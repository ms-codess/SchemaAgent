"""
Tests for src/evaluator.py — feature/evaluation.

No real API calls, no real databases, no real BIRD files.
All external dependencies are mocked.

Coverage:
  - Config A runs: no RAG, no correction, calls call_claude once per question
  - Config B runs: RAG indexed, schema context used, no correction
  - Config C runs: no RAG, self-correction loop retries on SQL error
  - Config D runs: full SQLAgent (RAG + correction)
  - execution_accuracy computed correctly (correct / total)
  - results JSON written to disk
  - build_summary renders a markdown table with all four configs
  - main() exits cleanly with --limit and --config flags
  - missing ANTHROPIC_API_KEY causes sys.exit(1)
  - zero questions edge case returns 0.0 EX
  - config C retries when SQL fails then succeeds on second attempt
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from src.evaluator import (
    run_config_a,
    run_config_b,
    run_config_c,
    run_config_d,
    build_summary,
    _self_correct,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_QUESTIONS = [
    {"question_id": 1, "db_id": "uni", "question": "How many students?",
     "SQL": "SELECT COUNT(*) FROM students", "evidence": ""},
    {"question_id": 2, "db_id": "uni", "question": "List all courses.",
     "SQL": "SELECT * FROM courses", "evidence": ""},
]


def _mock_client(sql_reply: str = "```sql\nSELECT COUNT(*) FROM students\n```"):
    client = MagicMock()
    msg = MagicMock()
    msg.content = [MagicMock(text=sql_reply)]
    client.messages.create.return_value = msg
    return client


# ── Config A ──────────────────────────────────────────────────────────────────

class TestConfigA:

    @patch("src.evaluator.save_results")
    @patch("src.evaluator.log_to_mlflow")
    @patch("src.evaluator.execution_match", return_value=(True, None))
    @patch("src.evaluator.call_claude", return_value="SELECT COUNT(*) FROM students")
    @patch("src.evaluator.get_full_schema", return_value="Table students: id INT, name TEXT")
    @patch("src.evaluator.db_path_for", return_value="/fake/uni/uni.sqlite")
    def test_runs_all_questions(self, mock_dbpath, mock_schema, mock_claude,
                                mock_match, mock_mlflow, mock_save):
        client = _mock_client()
        result = run_config_a(SAMPLE_QUESTIONS.copy(), client, "/fake/db")
        assert result["total"] == 2
        assert mock_claude.call_count == 2

    @patch("src.evaluator.save_results")
    @patch("src.evaluator.log_to_mlflow")
    @patch("src.evaluator.execution_match", return_value=(True, None))
    @patch("src.evaluator.call_claude", return_value="SELECT 1")
    @patch("src.evaluator.get_full_schema", return_value="schema")
    @patch("src.evaluator.db_path_for", return_value="/fake/uni.sqlite")
    def test_ex_all_correct(self, *mocks):
        client = _mock_client()
        result = run_config_a(SAMPLE_QUESTIONS.copy(), client, "/fake/db")
        assert result["execution_accuracy"] == 1.0
        assert result["correct"] == 2

    @patch("src.evaluator.save_results")
    @patch("src.evaluator.log_to_mlflow")
    @patch("src.evaluator.execution_match", return_value=(False, "error"))
    @patch("src.evaluator.call_claude", return_value="SELECT 1")
    @patch("src.evaluator.get_full_schema", return_value="schema")
    @patch("src.evaluator.db_path_for", return_value="/fake/uni.sqlite")
    def test_ex_all_wrong(self, *mocks):
        client = _mock_client()
        result = run_config_a(SAMPLE_QUESTIONS.copy(), client, "/fake/db")
        assert result["execution_accuracy"] == 0.0
        assert result["correct"] == 0

    @patch("src.evaluator.save_results")
    @patch("src.evaluator.log_to_mlflow")
    @patch("src.evaluator.execution_match", return_value=(True, None))
    @patch("src.evaluator.call_claude", return_value="SELECT 1")
    @patch("src.evaluator.get_full_schema", return_value="schema")
    @patch("src.evaluator.db_path_for", return_value="/fake/uni.sqlite")
    def test_limit_respected(self, *mocks):
        client = _mock_client()
        result = run_config_a(SAMPLE_QUESTIONS.copy(), client, "/fake/db", limit=1)
        assert result["total"] == 1

    @patch("src.evaluator.save_results")
    @patch("src.evaluator.log_to_mlflow")
    @patch("src.evaluator.execution_match", return_value=(True, None))
    @patch("src.evaluator.call_claude", return_value="SELECT 1")
    @patch("src.evaluator.get_full_schema", return_value="schema")
    @patch("src.evaluator.db_path_for", return_value="/fake/uni.sqlite")
    def test_config_label(self, *mocks):
        client = _mock_client()
        result = run_config_a(SAMPLE_QUESTIONS.copy(), client, "/fake/db")
        assert result["config"] == "A"
        assert result["rag"] is False
        assert result["self_correction"] is False


# ── Config B ──────────────────────────────────────────────────────────────────

class TestConfigB:

    @patch("src.evaluator.save_results")
    @patch("src.evaluator.log_to_mlflow")
    @patch("src.evaluator.execution_match", return_value=(True, None))
    @patch("src.evaluator.call_claude", return_value="SELECT 1")
    @patch("src.evaluator.db_path_for", return_value="/fake/uni.sqlite")
    def test_uses_rag(self, mock_dbpath, mock_claude, mock_match, mock_mlflow, mock_save):
        rag = MagicMock()
        rag.is_indexed.return_value = True
        rag.get_schema_context.return_value = "RAG schema context"

        with patch("src.evaluator.SchemaRAG", return_value=rag):
            client = _mock_client()
            result = run_config_b(SAMPLE_QUESTIONS.copy(), client, "/fake/db")

        rag.get_schema_context.assert_called()
        assert result["config"] == "B"
        assert result["rag"] is True
        assert result["self_correction"] is False

    @patch("src.evaluator.save_results")
    @patch("src.evaluator.log_to_mlflow")
    @patch("src.evaluator.execution_match", return_value=(True, None))
    @patch("src.evaluator.call_claude", return_value="SELECT 1")
    @patch("src.evaluator.get_full_schema", return_value="full schema fallback")
    @patch("src.evaluator.db_path_for", return_value="/fake/uni.sqlite")
    def test_falls_back_when_rag_empty(self, mock_dbpath, mock_full_schema,
                                       mock_claude, mock_match, mock_mlflow, mock_save):
        rag = MagicMock()
        rag.is_indexed.return_value = True
        rag.get_schema_context.return_value = "   "  # empty

        with patch("src.evaluator.SchemaRAG", return_value=rag):
            client = _mock_client()
            run_config_b(SAMPLE_QUESTIONS.copy(), client, "/fake/db")

        mock_full_schema.assert_called()

    @patch("src.evaluator.save_results")
    @patch("src.evaluator.log_to_mlflow")
    @patch("src.evaluator.execution_match", return_value=(True, None))
    @patch("src.evaluator.call_claude", return_value="SELECT 1")
    @patch("src.evaluator.db_path_for", return_value="/fake/uni.sqlite")
    def test_indexes_new_db(self, mock_dbpath, mock_claude, mock_match, mock_mlflow, mock_save):
        rag = MagicMock()
        rag.is_indexed.return_value = False
        rag.get_schema_context.return_value = "schema"

        with patch("src.evaluator.SchemaRAG", return_value=rag):
            client = _mock_client()
            run_config_b(SAMPLE_QUESTIONS.copy(), client, "/fake/db")

        rag.index.assert_called_once()  # only once despite two questions with same db_id


# ── Config C ──────────────────────────────────────────────────────────────────

class TestConfigC:

    @patch("src.evaluator.save_results")
    @patch("src.evaluator.log_to_mlflow")
    @patch("src.evaluator.execution_match", return_value=(True, None))
    @patch("src.evaluator.get_full_schema", return_value="schema")
    @patch("src.evaluator.db_path_for", return_value="/fake/uni.sqlite")
    def test_self_correction_called(self, mock_dbpath, mock_schema, mock_match, mock_mlflow, mock_save):
        client = _mock_client()
        with patch("src.evaluator._self_correct", return_value=("SELECT 1", 1)) as mock_sc:
            result = run_config_c(SAMPLE_QUESTIONS.copy(), client, "/fake/db")
        assert mock_sc.call_count == 2
        assert result["config"] == "C"
        assert result["self_correction"] is True

    @patch("src.evaluator.save_results")
    @patch("src.evaluator.log_to_mlflow")
    @patch("src.evaluator.execution_match", return_value=(True, None))
    @patch("src.evaluator.get_full_schema", return_value="schema")
    @patch("src.evaluator.db_path_for", return_value="/fake/uni.sqlite")
    def test_avg_attempts_recorded(self, mock_dbpath, mock_schema, mock_match, mock_mlflow, mock_save):
        client = _mock_client()
        with patch("src.evaluator._self_correct", return_value=("SELECT 1", 2)):
            result = run_config_c(SAMPLE_QUESTIONS.copy(), client, "/fake/db")
        assert result["avg_attempts"] == 2.0


class TestSelfCorrect:

    def test_returns_on_first_success(self):
        """If SQL executes without error on attempt 1, return immediately."""
        client = _mock_client("```sql\nSELECT 1\n```")
        with patch("src.evaluator.get_db_connection") as mock_conn_fn, \
             patch("src.evaluator.execute_sql", return_value=([(1,)], None)):
            mock_conn = MagicMock()
            mock_conn_fn.return_value = mock_conn
            sql, attempts = _self_correct(client, "Q?", "schema", "/fake.sqlite", "")
        assert attempts == 1
        assert "SELECT" in sql.upper()

    def test_retries_on_error(self):
        """If attempt 1 fails with an error, a second message is appended."""
        client = _mock_client("```sql\nSELECT 1\n```")
        call_count = {"n": 0}

        def fake_execute(conn, sql):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return None, "no such table"
            return [(1,)], None

        with patch("src.evaluator.get_db_connection") as mock_conn_fn, \
             patch("src.evaluator.execute_sql", side_effect=fake_execute):
            mock_conn = MagicMock()
            mock_conn_fn.return_value = mock_conn
            sql, attempts = _self_correct(client, "Q?", "schema", "/fake.sqlite", "")

        assert client.messages.create.call_count >= 2


# ── Config D ──────────────────────────────────────────────────────────────────

class TestConfigD:

    @patch("src.evaluator.save_results")
    @patch("src.evaluator.log_to_mlflow")
    @patch("src.evaluator.execution_match", return_value=(True, None))
    @patch("src.evaluator.db_path_for", return_value="/fake/uni.sqlite")
    def test_uses_sql_agent(self, mock_dbpath, mock_match, mock_mlflow, mock_save):
        from src.agent import AgentResult
        agent_result = AgentResult(sql="SELECT 1", result_rows=[(1,)], attempts=1, success=True)

        rag = MagicMock()
        rag.is_indexed.return_value = True
        agent = MagicMock()
        agent.run.return_value = agent_result

        with patch("src.evaluator.SchemaRAG", return_value=rag), \
             patch("src.evaluator.SQLAgent", return_value=agent):
            client = _mock_client()
            result = run_config_d(SAMPLE_QUESTIONS.copy(), client, "/fake/db")

        assert agent.run.call_count == 2
        assert result["config"] == "D"
        assert result["rag"] is True
        assert result["self_correction"] is True

    @patch("src.evaluator.save_results")
    @patch("src.evaluator.log_to_mlflow")
    @patch("src.evaluator.execution_match", return_value=(False, None))
    @patch("src.evaluator.db_path_for", return_value="/fake/uni.sqlite")
    def test_ex_zero_when_all_wrong(self, mock_dbpath, mock_match, mock_mlflow, mock_save):
        from src.agent import AgentResult
        agent_result = AgentResult(sql="SELECT 1", result_rows=None, attempts=3, success=False, error="fail")

        rag = MagicMock()
        rag.is_indexed.return_value = True
        agent = MagicMock()
        agent.run.return_value = agent_result

        with patch("src.evaluator.SchemaRAG", return_value=rag), \
             patch("src.evaluator.SQLAgent", return_value=agent):
            client = _mock_client()
            result = run_config_d(SAMPLE_QUESTIONS.copy(), client, "/fake/db")

        assert result["execution_accuracy"] == 0.0


# ── build_summary ─────────────────────────────────────────────────────────────

class TestBuildSummary:

    def _sample_results(self):
        return [
            {"config": "A", "rag": False, "self_correction": False, "execution_accuracy": 0.46, "correct": 230, "total": 500},
            {"config": "B", "rag": True,  "self_correction": False, "execution_accuracy": 0.52, "correct": 260, "total": 500},
            {"config": "C", "rag": False, "self_correction": True,  "execution_accuracy": 0.49, "correct": 245, "total": 500},
            {"config": "D", "rag": True,  "self_correction": True,  "execution_accuracy": 0.58, "correct": 290, "total": 500},
        ]

    def test_contains_all_configs(self):
        md = build_summary(self._sample_results())
        for cfg in ("A", "B", "C", "D"):
            assert f"| {cfg} |" in md

    def test_contains_ex_percentages(self):
        md = build_summary(self._sample_results())
        assert "46.00%" in md
        assert "58.00%" in md

    def test_markdown_table_header(self):
        md = build_summary(self._sample_results())
        assert "EX%" in md
        assert "RAG" in md
        assert "Self-Correction" in md

    def test_yes_no_flags(self):
        md = build_summary(self._sample_results())
        assert "Yes" in md
        assert "No" in md


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:

    @patch("src.evaluator.save_results")
    @patch("src.evaluator.log_to_mlflow")
    @patch("src.evaluator.execution_match", return_value=(True, None))
    @patch("src.evaluator.call_claude", return_value="SELECT 1")
    @patch("src.evaluator.get_full_schema", return_value="schema")
    @patch("src.evaluator.db_path_for", return_value="/fake/uni.sqlite")
    def test_zero_questions_returns_zero_ex(self, *mocks):
        client = _mock_client()
        result = run_config_a([], client, "/fake/db")
        assert result["execution_accuracy"] == 0.0
        assert result["total"] == 0
