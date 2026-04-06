"""
Tests for HybridFusion — feature/hybrid-fusion.

All tests mock the Claude API, SQLAgent, DocRAG, and IntentRouter so that
no real network calls or databases are needed.

Coverage:
  - database intent → SQL path only, answer from rows
  - document intent → doc path only, Claude synthesis
  - hybrid intent   → both paths, Claude fusion
  - hybrid intent, SQL fails → fused answer notes the failure
  - hybrid intent, no docs   → fused answer notes missing passages
  - database intent, no db_path → graceful empty AgentResult
  - single-cell SQL result returns value directly (no extra LLM call)
  - multi-row SQL result formatted correctly
  - FusionResult fields populated correctly for each intent
  - doc intent with no passages returns a user-friendly message
"""

import pytest
from dataclasses import dataclass
from typing import List, Optional
from unittest.mock import MagicMock, patch, PropertyMock

from src.fusion import HybridFusion, FusionResult
from src.agent import AgentResult
from src.router import RouteResult
from src.doc_rag import DocChunk


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_route(intent="database", confidence="high", reasoning="test"):
    return RouteResult(intent=intent, confidence=confidence, reasoning=reasoning)


def _make_agent_result(success=True, rows=None, sql="SELECT 1", attempts=1, error=None):
    return AgentResult(
        sql=sql,
        result_rows=rows if rows is not None else [("42",)],
        attempts=attempts,
        success=success,
        error=error,
    )


def _make_doc_chunks(texts=("Policy text here.",)):
    return [
        DocChunk(
            chunk_id=f"doc.txt::{i}",
            source=f"/docs/doc_{i}.txt",
            text=t,
            metadata={"source": f"/docs/doc_{i}.txt"},
        )
        for i, t in enumerate(texts)
    ]


def _make_fusion(intent="database", agent_result=None, doc_chunks=None, claude_reply="Fused answer."):
    """
    Build a HybridFusion instance with all dependencies mocked.
    Returns (fusion, mocks_dict).
    """
    client = MagicMock()
    # Mock Claude synthesis call
    msg_mock = MagicMock()
    msg_mock.content = [MagicMock(text=claude_reply)]
    client.messages.create.return_value = msg_mock

    schema_rag = MagicMock()
    doc_rag = MagicMock()
    router = MagicMock()

    router.classify.return_value = _make_route(intent=intent)

    if doc_chunks is None:
        doc_rag.retrieve_chunks.return_value = []
    else:
        doc_rag.retrieve_chunks.return_value = doc_chunks

    fusion = HybridFusion(
        client=client,
        schema_rag=schema_rag,
        doc_rag=doc_rag,
        router=router,
        model="claude-sonnet-4-5",
    )

    # Patch the internal SQLAgent.run
    if agent_result is not None:
        fusion._sql_agent = MagicMock()
        fusion._sql_agent.run.return_value = agent_result

    return fusion, {"client": client, "doc_rag": doc_rag, "router": router}


# ── Tests: database intent ────────────────────────────────────────────────────

class TestDatabaseIntent:

    def test_sql_path_only(self):
        """Database intent: doc pipeline not called."""
        ar = _make_agent_result(rows=[("5",)])
        fusion, mocks = _make_fusion(intent="database", agent_result=ar)
        result = fusion.answer("How many students?", db_id="uni", db_path="uni.sqlite")
        assert isinstance(result, FusionResult)
        mocks["doc_rag"].retrieve_chunks.assert_not_called()

    def test_single_cell_answer_no_llm(self):
        """Single-cell result returns value directly without calling Claude synthesis."""
        ar = _make_agent_result(rows=[("99",)])
        fusion, mocks = _make_fusion(intent="database", agent_result=ar)
        result = fusion.answer("Count?", db_id="db", db_path="db.sqlite")
        assert result.answer == "99"
        # Claude synthesis should NOT be called for database-only path
        mocks["client"].messages.create.assert_not_called()

    def test_multi_row_result_formatted(self):
        """Multi-row result is formatted as comma-separated lines."""
        rows = [("Alice", "3.8"), ("Bob", "3.5"), ("Carol", "3.2")]
        ar = _make_agent_result(rows=rows)
        fusion, _ = _make_fusion(intent="database", agent_result=ar)
        result = fusion.answer("List GPAs", db_id="db", db_path="db.sqlite")
        assert "Alice" in result.answer
        assert "Bob" in result.answer

    def test_sql_failure_answer(self):
        """Failed SQL returns user-friendly message."""
        ar = _make_agent_result(success=False, rows=None, error="table not found")
        fusion, _ = _make_fusion(intent="database", agent_result=ar)
        result = fusion.answer("Query?", db_id="db", db_path="db.sqlite")
        assert result.sql_success is False
        assert "table not found" in result.answer

    def test_result_fields_populated(self):
        """FusionResult fields are all set correctly for database intent."""
        ar = _make_agent_result(sql="SELECT COUNT(*) FROM t", rows=[(10,)], attempts=2)
        fusion, _ = _make_fusion(intent="database", agent_result=ar)
        result = fusion.answer("How many?", db_id="db", db_path="db.sqlite")
        assert result.intent == "database"
        assert result.sql == "SELECT COUNT(*) FROM t"
        assert result.sql_attempts == 2
        assert result.sql_success is True
        assert result.doc_passages is None

    def test_no_db_path_skips_sql(self):
        """If db_path is empty, SQL pipeline is skipped even for database intent."""
        fusion, mocks = _make_fusion(intent="database")
        fusion._sql_agent = MagicMock()
        result = fusion.answer("How many?", db_id="db", db_path="")
        fusion._sql_agent.run.assert_not_called()
        assert result.sql is None


# ── Tests: document intent ────────────────────────────────────────────────────

class TestDocumentIntent:

    def test_doc_path_only(self):
        """Document intent: SQL pipeline not called."""
        chunks = _make_doc_chunks(["The probation GPA threshold is 2.0."])
        fusion, mocks = _make_fusion(intent="document", doc_chunks=chunks, claude_reply="GPA is 2.0.")
        fusion._sql_agent = MagicMock()
        result = fusion.answer("What is the probation threshold?")
        fusion._sql_agent.run.assert_not_called()
        assert result.intent == "document"

    def test_doc_answer_from_claude(self):
        """Document intent calls Claude with doc passages."""
        chunks = _make_doc_chunks(["Probation requires GPA below 2.0."])
        fusion, mocks = _make_fusion(intent="document", doc_chunks=chunks, claude_reply="Below 2.0.")
        result = fusion.answer("Probation threshold?")
        assert result.answer == "Below 2.0."
        mocks["client"].messages.create.assert_called_once()

    def test_no_passages_returns_message(self):
        """Document intent with no indexed docs returns friendly message."""
        fusion, _ = _make_fusion(intent="document", doc_chunks=[])
        result = fusion.answer("What is the policy?")
        assert "No relevant document passages" in result.answer

    def test_result_fields(self):
        """FusionResult has no SQL fields for document intent."""
        chunks = _make_doc_chunks(["Policy text."])
        fusion, _ = _make_fusion(intent="document", doc_chunks=chunks, claude_reply="Answer.")
        result = fusion.answer("Policy question?")
        assert result.sql is None
        assert result.sql_rows is None
        assert result.doc_passages is not None
        assert len(result.sources) > 0

    def test_sources_populated(self):
        """Source filenames extracted from DocChunks."""
        chunks = _make_doc_chunks(["Text A.", "Text B."])
        fusion, _ = _make_fusion(intent="document", doc_chunks=chunks, claude_reply="Answer.")
        result = fusion.answer("Question?")
        assert len(result.sources) >= 1


# ── Tests: hybrid intent ──────────────────────────────────────────────────────

class TestHybridIntent:

    def test_both_pipelines_run(self):
        """Hybrid intent runs both SQL and doc pipelines."""
        ar = _make_agent_result(rows=[("Alice", "3.9")])
        chunks = _make_doc_chunks(["Graduation requires GPA ≥ 3.0."])
        fusion, mocks = _make_fusion(intent="hybrid", agent_result=ar, doc_chunks=chunks, claude_reply="Alice meets the requirement.")
        result = fusion.answer("Does Alice meet graduation requirements?", db_id="db", db_path="db.sqlite")
        fusion._sql_agent.run.assert_called_once()
        mocks["doc_rag"].retrieve_chunks.assert_called_once()
        assert result.answer == "Alice meets the requirement."

    def test_fused_answer_calls_claude(self):
        """Hybrid fusion calls Claude once with both sources."""
        ar = _make_agent_result(rows=[("3.9",)])
        chunks = _make_doc_chunks(["Min GPA is 3.0."])
        fusion, mocks = _make_fusion(intent="hybrid", agent_result=ar, doc_chunks=chunks, claude_reply="Yes, meets requirement.")
        result = fusion.answer("GPA check?", db_id="db", db_path="db.sqlite")
        mocks["client"].messages.create.assert_called_once()
        assert result.answer == "Yes, meets requirement."

    def test_hybrid_sql_failure_still_fuses(self):
        """Hybrid: even if SQL fails, Claude is called with failure info + doc passages."""
        ar = _make_agent_result(success=False, rows=None, error="no such table")
        chunks = _make_doc_chunks(["Policy states minimum wage is $15."])
        fusion, mocks = _make_fusion(intent="hybrid", agent_result=ar, doc_chunks=chunks, claude_reply="Cannot determine from DB.")
        result = fusion.answer("Check salary compliance?", db_id="db", db_path="db.sqlite")
        # Claude should still be called to fuse what's available
        mocks["client"].messages.create.assert_called_once()
        assert result.intent == "hybrid"

    def test_hybrid_no_docs_still_fuses(self):
        """Hybrid: no doc passages → Claude gets 'No document passages retrieved.'"""
        ar = _make_agent_result(rows=[("100",)])
        fusion, mocks = _make_fusion(intent="hybrid", agent_result=ar, doc_chunks=[], claude_reply="100 employees.")
        result = fusion.answer("How many below min wage?", db_id="db", db_path="db.sqlite")
        mocks["client"].messages.create.assert_called_once()
        # Verify the doc_passages string passed in contains the fallback message
        call_kwargs = mocks["client"].messages.create.call_args
        user_content = call_kwargs[1]["messages"][0]["content"]
        assert "No document passages retrieved" in user_content

    def test_hybrid_result_fields(self):
        """FusionResult has both SQL and doc fields for hybrid intent."""
        ar = _make_agent_result(sql="SELECT gpa FROM students WHERE name='Alice'", rows=[("3.9",)])
        chunks = _make_doc_chunks(["Threshold is 3.0."])
        fusion, _ = _make_fusion(intent="hybrid", agent_result=ar, doc_chunks=chunks, claude_reply="Meets it.")
        result = fusion.answer("Alice's GPA?", db_id="db", db_path="db.sqlite")
        assert result.sql is not None
        assert result.doc_passages is not None
        assert result.sql_success is True
        assert len(result.sources) > 0

    def test_route_metadata_preserved(self):
        """RouteResult confidence and reasoning are stored in FusionResult."""
        fusion, mocks = _make_fusion(intent="hybrid")
        mocks["router"].classify.return_value = RouteResult(
            intent="hybrid", confidence="medium", reasoning="needs both sources"
        )
        fusion._sql_agent = MagicMock()
        fusion._sql_agent.run.return_value = _make_agent_result()
        result = fusion.answer("Test?", db_id="db", db_path="db.sqlite")
        assert result.route_confidence == "medium"
        assert result.route_reasoning == "needs both sources"


# ── Tests: edge cases ─────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_more_than_20_rows_truncated(self):
        """Result with >20 rows shows first 20 + overflow note."""
        rows = [(str(i),) for i in range(25)]
        ar = _make_agent_result(rows=rows)
        fusion, _ = _make_fusion(intent="database", agent_result=ar)
        result = fusion.answer("All records?", db_id="db", db_path="db.sqlite")
        assert "5 more rows" in result.answer

    def test_empty_rows_message(self):
        """SQL succeeds but returns zero rows → clear message."""
        ar = AgentResult(sql="SELECT 1", result_rows=[], attempts=1, success=True, error=None)
        fusion, _ = _make_fusion(intent="database", agent_result=ar)
        result = fusion.answer("Empty query?", db_id="db", db_path="db.sqlite")
        assert "no rows" in result.answer.lower()

    def test_question_stored_in_result(self):
        """FusionResult.question matches the input."""
        ar = _make_agent_result()
        fusion, _ = _make_fusion(intent="database", agent_result=ar)
        q = "How many records are there?"
        result = fusion.answer(q, db_id="db", db_path="db.sqlite")
        assert result.question == q
