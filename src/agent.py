"""
SQL Agent — Branch 2 (feature/self-correction).

Receives a natural language question + db_id, retrieves schema context via
SchemaRAG, calls Claude Sonnet to generate SQL, executes it against SQLite,
and retries up to MAX_ATTEMPTS times if the SQL fails.

Three failure modes handled:
  syntax    — sqlglot rejects the SQL before it hits the database
  execution — SQLite raises an error at runtime
  empty     — query runs but returns zero rows

Usage:
    from src.schema import SchemaRAG
    import anthropic

    rag = SchemaRAG()
    client = anthropic.Anthropic()
    agent = SQLAgent(schema_rag=rag, client=client)
    result = agent.run(question="How many schools are there?",
                       db_id="california_schools",
                       db_path="data/bird/dev_databases/california_schools/california_schools.sqlite")
    print(result.sql, result.success, result.attempts)
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import sqlglot
import sqlglot.errors

from src.schema import SchemaRAG
from src.utils import get_db_connection, execute_sql, extract_sql_from_response
from baselines.runner import (
    SYSTEM_PROMPT,
    build_user_message,
    get_full_schema,
)

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3

# ── Correction prompt templates ───────────────────────────────────────────────

_SYNTAX_CORRECTION = (
    "Your SQL has a syntax error:\n{error}\n\n"
    "Fix it and return only the corrected SQL wrapped in ```sql ... ``` tags."
)

_EXECUTION_CORRECTION = (
    "SQLite returned this error when running your query:\n{error}\n\n"
    "Rewrite the query to fix this error. "
    "Return only the corrected SQL wrapped in ```sql ... ``` tags."
)

_EMPTY_CORRECTION = (
    "Your query ran successfully but returned no rows. "
    "Reconsider your table choices, JOIN conditions, and WHERE filters — "
    "there should be results for this question. "
    "Return only the corrected SQL wrapped in ```sql ... ``` tags."
)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class AgentResult:
    sql: str
    result_rows: Optional[List[Tuple]]
    attempts: int
    success: bool
    error: Optional[str] = field(default=None)


# ── Agent ─────────────────────────────────────────────────────────────────────

class SQLAgent:
    """
    Text-to-SQL agent with self-correction loop.

    Parameters
    ----------
    schema_rag : SchemaRAG
        Facade for schema retrieval — call get_schema_context() to get
        relevant table descriptions for a question.
    client : anthropic.Anthropic
        Authenticated Anthropic API client.
    model : str
        Claude model to use for SQL generation.
    max_attempts : int
        Maximum number of generation + correction attempts (default 3).
    """

    def __init__(
        self,
        schema_rag: SchemaRAG,
        client,
        model: str = "claude-sonnet-4-5",
        max_attempts: int = MAX_ATTEMPTS,
    ):
        self.schema_rag = schema_rag
        self.client = client
        self.model = model
        self.max_attempts = max_attempts

    # ── Public interface ──────────────────────────────────────────────────────

    def run(
        self,
        question: str,
        db_id: str,
        db_path: str,
        evidence: str = "",
    ) -> AgentResult:
        """
        Generate and execute SQL for a natural language question.

        Returns an AgentResult with the final SQL, result rows (if any),
        number of attempts used, success flag, and last error message.
        """
        schema_context = self._get_schema(question, db_id, db_path)
        used_full_schema = not self.schema_rag.is_indexed(db_id)
        messages = [
            {"role": "user", "content": build_user_message(question, schema_context, evidence)}
        ]

        last_sql = ""
        last_error: Optional[str] = None

        for attempt in range(1, self.max_attempts + 1):
            # Generate SQL
            reply_text = self._call_claude(messages)
            sql = extract_sql_from_response(reply_text)
            last_sql = sql

            # Step 1 — syntax check (no DB round-trip needed)
            syntax_err = self._check_syntax(sql)
            if syntax_err:
                last_error = syntax_err
                logger.debug("Attempt %d — syntax error: %s", attempt, syntax_err)
                if attempt < self.max_attempts:
                    messages = self._append_correction(
                        messages, reply_text,
                        _SYNTAX_CORRECTION.format(error=syntax_err),
                    )
                continue

            # Step 2 — execute against SQLite
            conn = get_db_connection(db_path)
            if conn is None:
                last_error = f"Could not open database: {db_path}"
                break

            rows, exec_err = execute_sql(conn, sql)
            conn.close()

            if exec_err:
                last_error = exec_err
                logger.debug("Attempt %d — execution error: %s", attempt, exec_err)
                if attempt < self.max_attempts:
                    # If RAG was used and we hit an execution error, the narrow
                    # schema context may be missing the required table. Escalate
                    # to the full schema and start a fresh prompt so Claude has
                    # complete visibility before the next attempt.
                    if not used_full_schema:
                        logger.info(
                            "Attempt %d — execution error with RAG context; "
                            "escalating to full schema for db=%s", attempt, db_id
                        )
                        full_schema = get_full_schema(db_path)
                        messages = [
                            {"role": "user", "content": build_user_message(question, full_schema, evidence)}
                        ]
                        used_full_schema = True
                    else:
                        messages = self._append_correction(
                            messages, reply_text,
                            _EXECUTION_CORRECTION.format(error=exec_err),
                        )
                continue

            if not rows:
                last_error = "empty_result"
                logger.debug("Attempt %d — empty result", attempt)
                if attempt < self.max_attempts:
                    messages = self._append_correction(
                        messages, reply_text, _EMPTY_CORRECTION,
                    )
                continue

            # Success
            logger.info("Question answered in %d attempt(s): %s", attempt, question[:60])
            return AgentResult(
                sql=sql,
                result_rows=rows,
                attempts=attempt,
                success=True,
                error=None,
            )

        # All attempts exhausted
        logger.warning(
            "Failed after %d attempt(s). Last error: %s | Question: %s",
            self.max_attempts, last_error, question[:60],
        )
        return AgentResult(
            sql=last_sql,
            result_rows=None,
            attempts=self.max_attempts,
            success=False,
            error=last_error,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_schema(self, question: str, db_id: str, db_path: str) -> str:
        """Retrieve schema context via RAG; fall back to full schema dump if empty."""
        if self.schema_rag.is_indexed(db_id):
            context = self.schema_rag.get_schema_context(question, db_id, use_rag=True)
            if context.strip():
                return context
            logger.warning("RAG returned empty context for db=%s, falling back", db_id)
        else:
            logger.info("db=%s not indexed — using full schema dump", db_id)
        return get_full_schema(db_path)

    def _call_claude(self, messages: list) -> str:
        """Single Claude API call; returns the raw reply text."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        return response.content[0].text

    def _check_syntax(self, sql: str) -> Optional[str]:
        """Return an error string if sql has a syntax error, else None."""
        if not sql.strip():
            return "Empty SQL returned"
        try:
            sqlglot.transpile(sql, read="sqlite", error_level=sqlglot.ErrorLevel.RAISE)
            return None
        except sqlglot.errors.SqlglotError as e:
            return str(e)

    @staticmethod
    def _append_correction(
        messages: list,
        assistant_reply: str,
        correction_text: str,
    ) -> list:
        """Append the assistant turn + correction user turn to the messages list."""
        return messages + [
            {"role": "assistant", "content": assistant_reply},
            {"role": "user", "content": correction_text},
        ]
