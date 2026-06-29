"""
Tests for baselines — runner utilities + baseline A/B/C smoke tests.

All tests are unit-level: they mock the Claude API and use the real
california_schools SQLite database so no BIRD JSON or API key is needed.
"""
import json
import os
import sqlite3
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.config import PROJECT_ROOT
from baselines.runner import (
    get_full_schema,
    db_path_for,
    execution_match,
    build_user_message,
    call_claude,
    save_results,
    SYSTEM_PROMPT,
)

DB_PATH = str(
    PROJECT_ROOT / "data" / "bird" / "dev_databases" / "california_schools" / "california_schools.sqlite"
)
DB_ID = "california_schools"
DB_ROOT = str(PROJECT_ROOT / "data" / "bird" / "dev_databases")

# Tests that require the real SQLite file are skipped if BIRD data isn't present
needs_bird_db = pytest.mark.skipif(
    not Path(DB_PATH).exists(),
    reason="BIRD data not present (data/ is gitignored — download separately)",
)


# ── runner: get_full_schema ───────────────────────────────────────────────────

@needs_bird_db
def test_get_full_schema_returns_text():
    schema = get_full_schema(DB_PATH)
    assert isinstance(schema, str)
    assert len(schema) > 0
    # Each line should look like "Table <name>: ..."
    lines = [l for l in schema.splitlines() if l.strip()]
    assert all(l.startswith("Table ") for l in lines)


def test_get_full_schema_missing_db():
    result = get_full_schema("/nonexistent/path/fake.sqlite")
    assert result == ""


# ── runner: db_path_for ───────────────────────────────────────────────────────

def test_db_path_for():
    path = db_path_for(DB_ID, DB_ROOT)
    assert path.endswith(f"{DB_ID}.sqlite")
    assert DB_ID in path


# ── runner: execution_match ───────────────────────────────────────────────────

@needs_bird_db
def test_execution_match_correct():
    gold = "SELECT COUNT(*) FROM schools"
    pred = "SELECT COUNT(*) FROM schools"
    match, err = execution_match(pred, gold, DB_PATH)
    assert match is True
    assert err is None


def test_execution_match_wrong_result():
    gold = "SELECT COUNT(*) FROM schools"
    pred = "SELECT 1"
    match, err = execution_match(pred, gold, DB_PATH)
    assert match is False


def test_execution_match_invalid_sql():
    gold = "SELECT COUNT(*) FROM schools"
    pred = "THIS IS NOT SQL"
    match, err = execution_match(pred, gold, DB_PATH)
    assert match is False
    assert err is not None


def test_execution_match_bad_db():
    match, err = execution_match("SELECT 1", "SELECT 1", "/nonexistent/db.sqlite")
    assert match is False
    assert err is not None


# ── runner: build_user_message ────────────────────────────────────────────────

def test_build_user_message_contains_question():
    msg = build_user_message("How many schools?", "Table schools: id TEXT, name TEXT")
    assert "How many schools?" in msg
    assert "Table schools" in msg


def test_build_user_message_with_evidence():
    msg = build_user_message("q", "schema", evidence="Use table schools")
    assert "Use table schools" in msg


def test_build_user_message_no_evidence():
    msg = build_user_message("q", "schema", evidence="")
    assert "Hint" not in msg


# ── runner: call_claude ───────────────────────────────────────────────────────

def _mock_client(sql: str) -> MagicMock:
    """Return a mock anthropic client whose messages.create returns sql."""
    client = MagicMock()
    response = MagicMock()
    response.content = [SimpleNamespace(text=f"```sql\n{sql}\n```")]
    client.messages.create.return_value = response
    return client


def test_call_claude_returns_sql():
    client = _mock_client("SELECT COUNT(*) FROM schools")
    result = call_claude(client, "How many schools?", "Table schools: id TEXT")
    assert "SELECT" in result.upper()
    client.messages.create.assert_called_once()


def test_call_claude_passes_system_prompt():
    client = _mock_client("SELECT 1")
    call_claude(client, "q", "schema")
    call_kwargs = client.messages.create.call_args
    assert call_kwargs.kwargs.get("system") == SYSTEM_PROMPT or \
           (call_kwargs.args and call_kwargs.args[0] == SYSTEM_PROMPT)


# ── runner: save_results ──────────────────────────────────────────────────────

def test_save_results_creates_file():
    data = [{"question": "q", "pred_sql": "SELECT 1", "match": True}]
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "sub", "results.json")
        save_results(data, out)
        assert Path(out).exists()
        loaded = json.loads(Path(out).read_text())
        assert loaded == data


# ── baseline_a smoke test ─────────────────────────────────────────────────────

def test_baseline_a_smoke(tmp_path):
    """Run baseline_a on 1 question with a mocked Claude client."""
    questions = [{
        "question_id": 1,
        "db_id": DB_ID,
        "question": "How many schools are there?",
        "SQL": "SELECT COUNT(*) FROM schools",
        "evidence": "",
    }]
    fake_sql = "SELECT COUNT(*) FROM schools"

    with patch("baselines.baseline_a.load_bird_questions", return_value=questions), \
         patch("baselines.baseline_a.anthropic.Anthropic") as MockAnthopic, \
         patch("baselines.baseline_a.RESULTS_PATH", str(tmp_path / "a.json")), \
         patch("baselines.runner.log_to_mlflow"), \
         patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
        MockAnthopic.return_value = _mock_client(fake_sql)
        from baselines.baseline_a import run
        result = run(limit=1)

    assert "execution_accuracy" in result
    assert result["total"] == 1
    assert 0.0 <= result["execution_accuracy"] <= 1.0


# ── baseline_b smoke test ─────────────────────────────────────────────────────

def test_baseline_b_smoke(tmp_path):
    """Run baseline_b on 1 question with mocked Claude + mocked RAG."""
    questions = [{
        "question_id": 1,
        "db_id": DB_ID,
        "question": "How many schools are there?",
        "SQL": "SELECT COUNT(*) FROM schools",
        "evidence": "",
    }]
    fake_sql = "SELECT COUNT(*) FROM schools"
    fake_schema = "Table schools: CDSCode TEXT, School TEXT"

    mock_rag = MagicMock()
    mock_rag.is_indexed.return_value = True
    mock_rag.get_schema_context.return_value = fake_schema

    with patch("baselines.baseline_b.load_bird_questions", return_value=questions), \
         patch("baselines.baseline_b.anthropic.Anthropic") as MockAnthropic, \
         patch("baselines.baseline_b.SchemaRAG", return_value=mock_rag), \
         patch("baselines.baseline_b.RESULTS_PATH", str(tmp_path / "b.json")), \
         patch("baselines.runner.log_to_mlflow"), \
         patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
        MockAnthropic.return_value = _mock_client(fake_sql)
        from baselines.baseline_b import run
        result = run(limit=1)

    assert "execution_accuracy" in result
    assert result["total"] == 1
    assert 0.0 <= result["execution_accuracy"] <= 1.0


# ── baseline_c smoke test ─────────────────────────────────────────────────────

def test_baseline_c_smoke(tmp_path):
    """Run baseline_c on 1 question with mocked Claude (SQL succeeds first try)."""
    questions = [{
        "question_id": 1,
        "db_id": DB_ID,
        "question": "How many schools are there?",
        "SQL": "SELECT COUNT(*) FROM schools",
        "evidence": "",
    }]
    fake_sql = "SELECT COUNT(*) FROM schools"

    with patch("baselines.baseline_c.load_bird_questions", return_value=questions), \
         patch("baselines.baseline_c.anthropic.Anthropic") as MockAnthropic, \
         patch("baselines.baseline_c.RESULTS_PATH", str(tmp_path / "c.json")), \
         patch("baselines.runner.log_to_mlflow"), \
         patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
        MockAnthropic.return_value = _mock_client(fake_sql)
        from baselines.baseline_c import run
        result = run(limit=1)

    assert "execution_accuracy" in result
    assert result["total"] == 1
    assert "avg_attempts" in result
    assert 0.0 <= result["execution_accuracy"] <= 1.0


@needs_bird_db
def test_baseline_c_retries_on_error(tmp_path):
    """Baseline C should retry when SQL fails, up to MAX_ATTEMPTS."""
    bad_sql = "SELECT * FROM nonexistent_table_xyz"
    good_sql = "SELECT COUNT(*) FROM schools"

    # First call returns bad SQL, second call returns good SQL
    client = MagicMock()
    responses = [
        MagicMock(content=[SimpleNamespace(text=f"```sql\n{bad_sql}\n```")]),
        MagicMock(content=[SimpleNamespace(text=f"```sql\n{good_sql}\n```")]),
    ]
    client.messages.create.side_effect = responses

    from baselines.baseline_c import call_with_correction
    sql, attempts = call_with_correction(client, "How many schools?", "Table schools: id TEXT", DB_PATH)

    assert attempts == 2
    assert "schools" in sql.lower()
    assert client.messages.create.call_count == 2
