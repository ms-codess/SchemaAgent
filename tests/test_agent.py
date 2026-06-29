"""
Tests for src/agent.py — SQLAgent with self-correction loop.

All Claude API calls and DB calls are mocked so no real SQLite file or API
key is needed. The DB path constant is kept for readability but the actual
file is never opened in these tests.
"""
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

import pytest

from src.agent import SQLAgent, AgentResult, MAX_ATTEMPTS
from src.config import PROJECT_ROOT

DB_PATH = "data/bird/dev_databases/california_schools/california_schools.sqlite"
DB_ID = "california_schools"

# Fake rows returned by a successful query
FAKE_ROWS = [(100,)]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_client(*sql_responses: str) -> MagicMock:
    """
    Return a mock Anthropic client whose messages.create cycles through
    sql_responses, wrapping each in ```sql ... ``` tags.
    """
    client = MagicMock()
    replies = [
        MagicMock(content=[SimpleNamespace(text=f"```sql\n{sql}\n```")])
        for sql in sql_responses
    ]
    client.messages.create.side_effect = replies
    return client


def _make_agent(client) -> SQLAgent:
    rag = MagicMock()
    rag.is_indexed.return_value = True
    rag.get_schema_context.return_value = "Table schools: CDSCode TEXT, School TEXT, County TEXT"
    return SQLAgent(schema_rag=rag, client=client)


def _mock_db(rows=FAKE_ROWS, error=None):
    """
    Return a (mock_conn, mock_execute_sql) pair.
    Patches src.agent.get_db_connection and src.agent.execute_sql.
    """
    mock_conn = MagicMock()
    return mock_conn, (rows, error)


# ── AgentResult dataclass ─────────────────────────────────────────────────────

def test_agent_result_fields():
    r = AgentResult(sql="SELECT 1", result_rows=[(1,)], attempts=1, success=True)
    assert r.sql == "SELECT 1"
    assert r.result_rows == [(1,)]
    assert r.attempts == 1
    assert r.success is True
    assert r.error is None


# ── Syntax checking ───────────────────────────────────────────────────────────

def test_check_syntax_valid():
    agent = _make_agent(_make_client("SELECT 1"))
    assert agent._check_syntax("SELECT COUNT(*) FROM schools") is None


def test_check_syntax_invalid():
    agent = _make_agent(_make_client("SELECT 1"))
    err = agent._check_syntax("SELECT * FORM schools")
    assert err is not None
    assert isinstance(err, str)


def test_check_syntax_empty_sql():
    agent = _make_agent(_make_client("SELECT 1"))
    err = agent._check_syntax("")
    assert err is not None


# ── One-shot success ──────────────────────────────────────────────────────────

def test_run_one_shot_success():
    client = _make_client("SELECT COUNT(*) FROM schools")
    agent = _make_agent(client)
    with patch("src.agent.get_db_connection") as mock_conn, \
         patch("src.agent.execute_sql", return_value=(FAKE_ROWS, None)):
        mock_conn.return_value = MagicMock()
        result = agent.run("How many schools?", DB_ID, DB_PATH)

    assert result.success is True
    assert result.attempts == 1
    assert result.error is None
    assert result.result_rows == FAKE_ROWS
    client.messages.create.assert_called_once()


# ── Syntax error → retry → success ───────────────────────────────────────────

def test_run_syntax_error_then_success():
    client = _make_client(
        "SELECT * FORM schools",          # attempt 1 — syntax error
        "SELECT COUNT(*) FROM schools",   # attempt 2 — success
    )
    agent = _make_agent(client)
    with patch("src.agent.get_db_connection") as mock_conn, \
         patch("src.agent.execute_sql", return_value=(FAKE_ROWS, None)):
        mock_conn.return_value = MagicMock()
        result = agent.run("How many schools?", DB_ID, DB_PATH)

    assert result.success is True
    assert result.attempts == 2
    assert client.messages.create.call_count == 2

    # Second call must have more messages than the first (correction turns appended)
    first_msg_count = len(client.messages.create.call_args_list[0][1]["messages"])
    second_msg_count = len(client.messages.create.call_args_list[1][1]["messages"])
    assert second_msg_count > first_msg_count


# ── Execution error → retry → success ────────────────────────────────────────

def test_run_execution_error_then_success():
    client = _make_client(
        "SELECT COUNT(*) FROM nonexistent_table_xyz",  # attempt 1 — execution error
        "SELECT COUNT(*) FROM schools",                 # attempt 2 — success
    )
    agent = _make_agent(client)
    exec_results = [
        (None, "no such table: nonexistent_table_xyz"),  # attempt 1 fails
        (FAKE_ROWS, None),                               # attempt 2 succeeds
    ]
    full_schema = "Table schools: CDSCode TEXT, School TEXT"
    with patch("src.agent.get_db_connection") as mock_conn, \
         patch("src.agent.execute_sql", side_effect=exec_results), \
         patch("src.agent.get_full_schema", return_value=full_schema) as mock_full:
        mock_conn.return_value = MagicMock()
        result = agent.run("How many schools?", DB_ID, DB_PATH)

    assert result.success is True
    assert result.attempts == 2
    assert client.messages.create.call_count == 2

    # On execution error with RAG context, agent resets to a fresh single-message
    # prompt using the full schema — NOT an appended correction turn.
    mock_full.assert_called_once_with(DB_PATH)
    second_call_messages = client.messages.create.call_args_list[1][1]["messages"]
    assert len(second_call_messages) == 1
    # Content may be a plain string or a cached list — extract text either way.
    raw = second_call_messages[0]["content"]
    content_text = raw[0]["text"] if isinstance(raw, list) else raw
    assert full_schema in content_text


# ── Empty result → retry → success ───────────────────────────────────────────

def test_run_empty_result_then_success():
    client = _make_client(
        "SELECT * FROM schools WHERE School = 'FAKE'",
        "SELECT COUNT(*) FROM schools",
    )
    agent = _make_agent(client)
    exec_results = [
        ([], None),          # attempt 1 — empty result
        (FAKE_ROWS, None),   # attempt 2 — success
    ]
    with patch("src.agent.get_db_connection") as mock_conn, \
         patch("src.agent.execute_sql", side_effect=exec_results):
        mock_conn.return_value = MagicMock()
        result = agent.run("How many schools?", DB_ID, DB_PATH)

    assert result.success is True
    assert result.attempts == 2

    second_call_messages = client.messages.create.call_args_list[1].kwargs["messages"]

    def _text(msg):
        raw = msg["content"]
        return raw[0]["text"] if isinstance(raw, list) else raw

    assert any(
        "no rows" in _text(m).lower() or "empty" in _text(m).lower()
        for m in second_call_messages
        if m["role"] == "user"
    )


# ── Max attempts cap ──────────────────────────────────────────────────────────

def test_run_stops_at_max_attempts():
    bad_sql = "SELECT * FORM schools"  # syntax error every time
    client = _make_client(bad_sql, bad_sql, bad_sql)
    agent = _make_agent(client)
    result = agent.run("How many schools?", DB_ID, DB_PATH)

    assert result.success is False
    assert result.attempts == MAX_ATTEMPTS
    assert client.messages.create.call_count == MAX_ATTEMPTS


def test_run_never_exceeds_max_attempts_on_exec_error():
    bad_sql = "SELECT COUNT(*) FROM no_such_table_abc"
    client = _make_client(bad_sql, bad_sql, bad_sql)
    agent = _make_agent(client)
    with patch("src.agent.get_db_connection") as mock_conn, \
         patch("src.agent.execute_sql", return_value=(None, "no such table: no_such_table_abc")):
        mock_conn.return_value = MagicMock()
        result = agent.run("How many schools?", DB_ID, DB_PATH)

    assert result.success is False
    assert result.attempts == MAX_ATTEMPTS
    assert client.messages.create.call_count == MAX_ATTEMPTS


# ── Result contains SQL even on failure ──────────────────────────────────────

def test_run_failed_result_contains_last_sql():
    bad_sql = "SELECT * FORM schools"
    client = _make_client(bad_sql, bad_sql, bad_sql)
    agent = _make_agent(client)
    result = agent.run("q", DB_ID, DB_PATH)

    assert result.sql != ""
    assert result.result_rows is None
    assert result.error is not None


# ── Schema fallback ───────────────────────────────────────────────────────────

def test_run_falls_back_to_full_schema_when_not_indexed():
    client = _make_client("SELECT COUNT(*) FROM schools")
    rag = MagicMock()
    rag.is_indexed.return_value = False
    agent = SQLAgent(schema_rag=rag, client=client)

    with patch("src.agent.get_db_connection") as mock_conn, \
         patch("src.agent.execute_sql", return_value=(FAKE_ROWS, None)), \
         patch("src.agent.get_full_schema", return_value="Table schools: id TEXT"):
        mock_conn.return_value = MagicMock()
        result = agent.run("How many schools?", DB_ID, DB_PATH)

    rag.get_schema_context.assert_not_called()
    assert result.success is True


def test_run_falls_back_when_rag_returns_empty():
    client = _make_client("SELECT COUNT(*) FROM schools")
    rag = MagicMock()
    rag.is_indexed.return_value = True
    rag.get_schema_context.return_value = ""
    agent = SQLAgent(schema_rag=rag, client=client)

    with patch("src.agent.get_db_connection") as mock_conn, \
         patch("src.agent.execute_sql", return_value=(FAKE_ROWS, None)), \
         patch("src.agent.get_full_schema", return_value="Table schools: id TEXT"):
        mock_conn.return_value = MagicMock()
        result = agent.run("How many schools?", DB_ID, DB_PATH)

    assert result.success is True


# ── Multi-turn message structure ──────────────────────────────────────────────

def test_correction_appends_two_messages():
    messages = [{"role": "user", "content": "original question"}]
    updated = SQLAgent._append_correction(messages, "assistant reply", "correction text")

    assert len(updated) == 3
    assert updated[0] == messages[0]
    assert updated[1] == {"role": "assistant", "content": "assistant reply"}
    assert updated[2] == {"role": "user", "content": "correction text"}
    # original list must not be mutated
    assert len(messages) == 1
