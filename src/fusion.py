"""
Hybrid Fusion — feature/hybrid-fusion.

Takes a natural language question and routes it through one of three paths:
  database  — SQLAgent generates + executes SQL, returns structured rows
  document  — DocRAG retrieves relevant passages, Claude synthesises an answer
  hybrid    — BOTH paths run, Claude fuses the SQL result and doc passages
              into a single coherent answer

This is the novel contribution of the thesis: no prior Text-to-SQL system
formally evaluates fused structured + unstructured retrieval on BIRD.

Usage:
    import anthropic
    from src.schema import SchemaRAG
    from src.doc_rag import DocRAG
    from src.router import IntentRouter
    from src.fusion import HybridFusion

    client = anthropic.Anthropic()
    fusion = HybridFusion(
        client=client,
        schema_rag=SchemaRAG(),
        doc_rag=DocRAG(),
        router=IntentRouter(client),
    )
    result = fusion.answer(
        question="Does Alice's GPA meet the academic probation threshold?",
        db_id="university",
        db_path="data/bird/dev_databases/university/university.sqlite",
    )
    print(result.answer)
    print(result.intent, result.sources)
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from src.agent import SQLAgent, AgentResult
from src.doc_rag import DocRAG
from src.router import IntentRouter, RouteResult
from src.schema import SchemaRAG

logger = logging.getLogger(__name__)

# ── Fusion prompt ─────────────────────────────────────────────────────────────

_FUSION_SYSTEM = """\
You are a helpful assistant that answers questions by combining evidence from
two sources: a SQL database result and policy/document passages.

Your job is to synthesise both pieces of evidence into a single, direct answer.
- Be concise — one to three sentences.
- Cite the specific numbers from the database result where relevant.
- Reference the policy/document where relevant.
- If the sources contradict each other, say so clearly.
- Do not add information that is not present in the provided evidence.
"""

_FUSION_USER = """\
Question: {question}

DATABASE RESULT (SQL query output):
{db_result}

DOCUMENT PASSAGES:
{doc_passages}

Using ONLY the evidence above, answer the question concisely.
"""

# ── Document-only synthesis prompt ───────────────────────────────────────────

_DOC_SYSTEM = """\
You are a helpful assistant that answers questions from policy and document passages.
Be concise — one to three sentences. Only use the provided passages; do not add
information that is not present.
"""

_DOC_USER = """\
Question: {question}

DOCUMENT PASSAGES:
{doc_passages}

Answer concisely using only the passages above.
"""


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class FusionResult:
    question: str
    intent: str                          # "database" | "document" | "hybrid"
    answer: str                          # final natural language answer
    sql: Optional[str] = None            # generated SQL (database / hybrid only)
    sql_rows: Optional[list] = None      # raw rows returned by the DB
    doc_passages: Optional[str] = None   # raw passages retrieved from docs
    route_confidence: str = "high"       # from IntentRouter
    route_reasoning: str = ""            # from IntentRouter
    sql_attempts: int = 0                # how many SQL generation attempts
    sql_success: bool = False            # did SQL execute successfully
    sources: List[str] = field(default_factory=list)  # filenames of doc sources


# ── HybridFusion ──────────────────────────────────────────────────────────────

class HybridFusion:
    """
    Unified answer layer that routes questions, runs the appropriate retrieval
    pipelines, and fuses the results into a single natural language answer.

    Parameters
    ----------
    client : anthropic.Anthropic
        Authenticated Anthropic API client.
    schema_rag : SchemaRAG
        Schema retrieval facade (used inside SQLAgent).
    doc_rag : DocRAG
        Document retrieval instance.
    router : IntentRouter
        Intent classifier — decides which pipeline(s) to run.
    model : str
        Claude model used for fusion and doc synthesis.
    """

    def __init__(
        self,
        client,
        schema_rag: SchemaRAG,
        doc_rag: DocRAG,
        router: IntentRouter,
        model: str = "claude-sonnet-4-5",
    ):
        self.client = client
        self.doc_rag = doc_rag
        self.router = router
        self.model = model
        self._sql_agent = SQLAgent(schema_rag=schema_rag, client=client, model=model)

    # ── Public interface ──────────────────────────────────────────────────────

    def answer(
        self,
        question: str,
        db_id: str = "",
        db_path: str = "",
        evidence: str = "",
        use_hyde: bool = False,
        top_k_docs: int = 3,
    ) -> FusionResult:
        """
        Route, retrieve, and fuse an answer for a natural language question.

        Parameters
        ----------
        question : str
            The user's question.
        db_id : str
            BIRD database identifier (e.g. "california_schools").
        db_path : str
            Path to the SQLite file.
        evidence : str
            Optional external hint injected into the SQL prompt (BIRD evidence field).
        use_hyde : bool
            Use HyDE for document retrieval (better quality, costs one extra API call).
        top_k_docs : int
            Number of document passages to retrieve.

        Returns
        -------
        FusionResult with answer, SQL, passages, and all diagnostic fields.
        """
        route: RouteResult = self.router.classify(question)
        intent = route.intent
        logger.info("Routing %r → %s (%s)", question[:60], intent, route.confidence)

        agent_result: Optional[AgentResult] = None
        doc_passages: str = ""
        doc_sources: List[str] = []

        # ── Run SQL pipeline ──────────────────────────────────────────────────
        if intent in ("database", "hybrid") and db_path:
            agent_result = self._run_sql(question, db_id, db_path, evidence)

        # ── Run document pipeline ─────────────────────────────────────────────
        if intent in ("document", "hybrid"):
            doc_passages, doc_sources = self._run_docs(
                question, use_hyde=use_hyde, top_k=top_k_docs
            )

        # ── Synthesise answer ─────────────────────────────────────────────────
        answer = self._synthesise(intent, question, agent_result, doc_passages)

        return FusionResult(
            question=question,
            intent=intent,
            answer=answer,
            sql=agent_result.sql if agent_result else None,
            sql_rows=agent_result.result_rows if agent_result else None,
            doc_passages=doc_passages or None,
            route_confidence=route.confidence,
            route_reasoning=route.reasoning,
            sql_attempts=agent_result.attempts if agent_result else 0,
            sql_success=agent_result.success if agent_result else False,
            sources=doc_sources,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _run_sql(
        self, question: str, db_id: str, db_path: str, evidence: str
    ) -> AgentResult:
        """Run the SQL agent; always returns an AgentResult (may have success=False)."""
        logger.debug("Running SQL pipeline for: %s", question[:60])
        return self._sql_agent.run(
            question=question,
            db_id=db_id,
            db_path=db_path,
            evidence=evidence,
        )

    def _run_docs(
        self, question: str, use_hyde: bool, top_k: int
    ):
        """Retrieve document passages; returns (passages_str, source_filenames)."""
        logger.debug("Running doc pipeline for: %s", question[:60])
        chunks = self.doc_rag.retrieve_chunks(
            question=question,
            top_k=top_k,
            use_hyde=use_hyde,
            hyde_client=self.client if use_hyde else None,
        )
        if not chunks:
            return "", []

        from pathlib import Path
        passages = "\n\n---\n\n".join(
            f"[Source: {Path(c.source).name}]\n{c.text}" for c in chunks
        )
        sources = list({Path(c.source).name for c in chunks})
        return passages, sources

    def _synthesise(
        self,
        intent: str,
        question: str,
        agent_result: Optional[AgentResult],
        doc_passages: str,
    ) -> str:
        """Call Claude to produce the final answer given intent and available evidence."""

        if intent == "database":
            return self._answer_from_db(agent_result)

        if intent == "document":
            if not doc_passages:
                return "No relevant document passages were found to answer this question."
            return self._call_claude(
                system=_DOC_SYSTEM,
                user=_DOC_USER.format(
                    question=question,
                    doc_passages=doc_passages,
                ),
            )

        # hybrid
        db_result_str = self._format_db_result(agent_result)
        doc_str = doc_passages or "No document passages retrieved."
        return self._call_claude(
            system=_FUSION_SYSTEM,
            user=_FUSION_USER.format(
                question=question,
                db_result=db_result_str,
                doc_passages=doc_str,
            ),
        )

    @staticmethod
    def _answer_from_db(agent_result: Optional[AgentResult]) -> str:
        """Format a plain-text answer from raw SQL rows (no extra LLM call needed)."""
        if agent_result is None or not agent_result.success:
            error = agent_result.error if agent_result else "no result"
            return f"The database query did not return a result. ({error})"
        rows = agent_result.result_rows or []
        if not rows:
            return "The query returned no rows."
        # Return a compact representation; the UI layer can format further
        if len(rows) == 1 and len(rows[0]) == 1:
            return str(rows[0][0])
        lines = [", ".join(str(v) for v in row) for row in rows[:20]]
        suffix = f"\n(... {len(rows) - 20} more rows)" if len(rows) > 20 else ""
        return "\n".join(lines) + suffix

    @staticmethod
    def _format_db_result(agent_result: Optional[AgentResult]) -> str:
        """Describe the DB result for injection into the fusion prompt."""
        if agent_result is None:
            return "No database query was executed."
        if not agent_result.success:
            return f"Database query failed: {agent_result.error}"
        rows = agent_result.result_rows or []
        if not rows:
            return "Query succeeded but returned no rows."
        lines = [", ".join(str(v) for v in row) for row in rows[:10]]
        suffix = f"\n(... {len(rows) - 10} more rows)" if len(rows) > 10 else ""
        return f"SQL: {agent_result.sql}\nRows:\n" + "\n".join(lines) + suffix

    def _call_claude(self, system: str, user: str) -> str:
        """Single Claude API call for synthesis."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text.strip()
