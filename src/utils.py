import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def load_bird_questions(dev_json_path: str) -> List[Dict[str, Any]]:
    """
    Read a BIRD dev.json file and return a list of dictionaries with
    keys: db_id, question, SQL, evidence.
    The input JSON can be either a list of question objects or a dict
    containing a list under common keys such as "data" or "questions".
    """
    path = Path(dev_json_path)
    if not path.exists():
        raise FileNotFoundError(f"dev.json not found at {dev_json_path}")

    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = (
            payload.get("data")
            or payload.get("questions")
            or payload.get("examples")
        )
    else:
        records = None

    if records is None or not isinstance(records, list):
        raise ValueError("dev.json must contain a list of question objects")

    normalized: List[Dict[str, Any]] = []
    for entry in records:
        if not isinstance(entry, dict):
            continue

        db_id = entry.get("db_id")
        question = entry.get("question") or entry.get("text") or entry.get("utterance")
        sql_value = entry.get("SQL") if "SQL" in entry else entry.get("sql") or entry.get("query")
        evidence = entry.get("evidence") if "evidence" in entry else entry.get("evidence_text") or entry.get("evidence_list")

        normalized.append(
            {
                "db_id": db_id,
                "question": question,
                "SQL": sql_value,
                "evidence": evidence,
            }
        )

    return normalized


def get_db_connection(db_path: str) -> Optional[sqlite3.Connection]:
    """
    Open a read-only SQLite connection to the database at db_path.
    Returns None if the file does not exist or the connection fails.
    """
    path = Path(db_path)
    if not path.exists():
        return None

    try:
        uri = f"file:{path.as_posix()}?mode=ro"
        return sqlite3.connect(uri, uri=True, check_same_thread=False)
    except sqlite3.Error:
        return None


def execute_sql(connection: Optional[sqlite3.Connection], sql: str) -> Tuple[Optional[List[Tuple[Any, ...]]], Optional[str]]:
    """
    Execute SQL against the provided SQLite connection.
    Returns (results, None) on success or (None, error_message) on failure.
    Never raises an exception.
    """
    if connection is None:
        return None, "No database connection provided."

    cursor = connection.cursor()
    try:
        cursor.execute(sql)
        rows = cursor.fetchall()
        return rows, None
    except sqlite3.Error as exc:  # broad but intentional to avoid raising
        return None, str(exc)
    finally:
        cursor.close()


_CODE_BLOCK_RE = re.compile(r"```(?:sql)?\s*([\s\S]*?)```", re.IGNORECASE)
_SQL_FALLBACK_RE = re.compile(
    r"\b(SELECT|WITH|INSERT|UPDATE|DELETE|CREATE|DROP)\b[\s\S]*",
    re.IGNORECASE,
)


def extract_sql_from_response(text: str) -> str:
    """
    Extract SQL wrapped in a markdown code block; fall back to the first
    SQL-looking statement if no code block is present.
    Returns an empty string when nothing plausible is found.
    """
    if not text:
        return ""

    block_match = _CODE_BLOCK_RE.search(text)
    if block_match:
        return block_match.group(1).strip()

    fallback_match = _SQL_FALLBACK_RE.search(text)
    if fallback_match:
        candidate = fallback_match.group(0)
        candidate = candidate.split("```", 1)[0]
        return candidate.strip()

    return ""


# ── SQL prompt helpers ────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert SQLite developer. "
    "Given a database schema and a natural language question, write a single valid SQLite SELECT query. "
    "Return ONLY the SQL query wrapped in ```sql ... ``` tags. No explanation."
)


def build_user_message(question: str, schema_context: str, evidence: str = "") -> str:
    parts = [f"Database schema:\n{schema_context}", f"Question: {question}"]
    if evidence:
        parts.append(f"Hint: {evidence}")
    parts.append("Write the SQL query.")
    return "\n\n".join(parts)


def get_full_schema(db_path: str) -> str:
    """Return a plain-text dump of every table and column in the database."""
    conn = get_db_connection(db_path)
    if conn is None:
        return ""
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cursor.fetchall() if r[0] and not r[0].startswith("sqlite_")]
        lines = []
        for table in tables:
            cursor.execute(f'PRAGMA table_info("{table}")')
            cols = cursor.fetchall()
            col_defs = ", ".join(f"{c[1]} {c[2]}" for c in cols if c[1])
            lines.append(f"Table {table}: {col_defs}")
        return "\n".join(lines)
    finally:
        conn.close()
