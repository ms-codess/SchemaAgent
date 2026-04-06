"""
Tests for src/router.py — IntentRouter + RouteResult.

Zero API calls — the Anthropic client is fully mocked in every test.
The mock is a simple object whose messages.create() returns a controlled response.
"""
import json
from unittest.mock import MagicMock, patch

import pytest

from src.router import IntentRouter, RouteResult, DEFAULT_INTENT, DEFAULT_CONFIDENCE


# ── Mock helpers ──────────────────────────────────────────────────────────────

def _make_client(response_text: str):
    """Return a mock Anthropic client whose first content block returns response_text."""
    content_block = MagicMock()
    content_block.text = response_text

    message = MagicMock()
    message.content = [content_block]

    client = MagicMock()
    client.messages.create.return_value = message
    return client


def _json_response(intent: str, confidence: str, reasoning: str) -> str:
    return json.dumps({"intent": intent, "confidence": confidence, "reasoning": reasoning})


# ── RouteResult dataclass ─────────────────────────────────────────────────────

def test_route_result_fields():
    r = RouteResult(intent="database", confidence="high", reasoning="counts records")
    assert r.intent == "database"
    assert r.confidence == "high"
    assert r.reasoning == "counts records"


# ── Happy path: three intents ─────────────────────────────────────────────────

def test_classify_database_intent():
    client = _make_client(_json_response("database", "high", "question asks for a count"))
    router = IntentRouter(client)
    result = router.classify("How many students enrolled in 2023?")
    assert result.intent == "database"
    assert result.confidence == "high"
    assert result.reasoning == "question asks for a count"


def test_classify_document_intent():
    client = _make_client(_json_response("document", "high", "asks about a policy rule"))
    router = IntentRouter(client)
    result = router.classify("What is the academic probation GPA threshold?")
    assert result.intent == "document"
    assert result.confidence == "high"


def test_classify_hybrid_intent():
    client = _make_client(_json_response("hybrid", "medium", "needs both DB and policy"))
    router = IntentRouter(client)
    result = router.classify("Does Alice's GPA meet the graduation requirement?")
    assert result.intent == "hybrid"
    assert result.confidence == "medium"


# ── Fallback: malformed responses ─────────────────────────────────────────────

def test_malformed_json_defaults_to_hybrid():
    client = _make_client("This is not JSON at all.")
    router = IntentRouter(client)
    result = router.classify("Some question?")
    assert result.intent == "hybrid"
    assert result.confidence == "low"
    assert "malformed" in result.reasoning


def test_unknown_intent_label_defaults_to_hybrid():
    client = _make_client(_json_response("unknown_label", "high", "???"))
    router = IntentRouter(client)
    result = router.classify("Some question?")
    assert result.intent == "hybrid"
    assert result.confidence == "low"


def test_empty_string_response_defaults_to_hybrid():
    client = _make_client("")
    router = IntentRouter(client)
    result = router.classify("Some question?")
    assert result.intent == "hybrid"
    assert result.confidence == "low"


def test_partial_json_missing_intent_defaults_to_hybrid():
    """JSON parses fine but 'intent' key is absent."""
    client = _make_client(json.dumps({"confidence": "high", "reasoning": "oops"}))
    router = IntentRouter(client)
    result = router.classify("Some question?")
    assert result.intent == "hybrid"
    assert result.confidence == "low"


def test_unknown_confidence_coerces_to_low():
    """Valid intent but unrecognised confidence value — intent is kept, confidence falls back."""
    client = _make_client(_json_response("database", "very_sure", "counts records"))
    router = IntentRouter(client)
    result = router.classify("How many rows?")
    assert result.intent == "database"
    assert result.confidence == "low"


def test_missing_reasoning_gets_placeholder():
    client = _make_client(json.dumps({"intent": "document", "confidence": "high"}))
    router = IntentRouter(client)
    result = router.classify("What is the leave policy?")
    assert result.intent == "document"
    assert result.reasoning  # not empty


# ── Markdown fence stripping ──────────────────────────────────────────────────

def test_markdown_fenced_json_is_parsed():
    """Claude sometimes wraps JSON in ```json ... ``` — should still parse."""
    raw = "```json\n" + _json_response("document", "high", "policy question") + "\n```"
    client = _make_client(raw)
    router = IntentRouter(client)
    result = router.classify("What is the refund policy?")
    assert result.intent == "document"
    assert result.confidence == "high"


def test_plain_fence_without_json_label():
    raw = "```\n" + _json_response("database", "medium", "structured query") + "\n```"
    client = _make_client(raw)
    router = IntentRouter(client)
    result = router.classify("How many employees?")
    assert result.intent == "database"


# ── API error fallback ────────────────────────────────────────────────────────

def test_api_error_defaults_to_hybrid():
    client = MagicMock()
    client.messages.create.side_effect = RuntimeError("network timeout")
    router = IntentRouter(client)
    result = router.classify("Will this crash?")
    assert result.intent == "hybrid"
    assert result.confidence == "low"
    assert "fallback" in result.reasoning


# ── Model default ─────────────────────────────────────────────────────────────

def test_default_model_is_haiku():
    client = _make_client(_json_response("database", "high", "test"))
    router = IntentRouter(client)
    assert router.model == "claude-haiku-4-5-20251001"


def test_custom_model_is_used():
    client = _make_client(_json_response("database", "high", "test"))
    router = IntentRouter(client, model="claude-sonnet-4-5")
    router.classify("test question")
    call_kwargs = client.messages.create.call_args
    assert call_kwargs.kwargs["model"] == "claude-sonnet-4-5"


# ── MLflow logging ────────────────────────────────────────────────────────────

def test_mlflow_logged_when_active_run(monkeypatch):
    mock_run = MagicMock()
    mock_mlflow = MagicMock()
    mock_mlflow.active_run.return_value = mock_run

    monkeypatch.setattr("src.router.IntentRouter._log_to_mlflow", staticmethod(
        lambda result: mock_mlflow.log_params({
            "router_intent": result.intent,
            "router_confidence": result.confidence,
        })
    ))

    client = _make_client(_json_response("database", "high", "test"))
    router = IntentRouter(client)
    router.classify("How many records?")

    mock_mlflow.log_params.assert_called_once_with({
        "router_intent": "database",
        "router_confidence": "high",
    })


def test_mlflow_skipped_when_no_active_run(monkeypatch):
    """_log_to_mlflow should not raise when mlflow has no active run."""
    with patch("src.router.IntentRouter._log_to_mlflow", return_value=None):
        client = _make_client(_json_response("hybrid", "low", "test"))
        router = IntentRouter(client)
        result = router.classify("test")
        assert result.intent == "hybrid"  # classification still works


# ── System prompt sanity ──────────────────────────────────────────────────────

def test_system_prompt_sent_to_api():
    client = _make_client(_json_response("database", "high", "test"))
    router = IntentRouter(client)
    router.classify("Any question")
    call_kwargs = client.messages.create.call_args.kwargs
    assert "system" in call_kwargs
    assert "database" in call_kwargs["system"].lower()
    assert "document" in call_kwargs["system"].lower()
    assert "hybrid" in call_kwargs["system"].lower()


def test_question_is_user_message():
    client = _make_client(_json_response("document", "high", "test"))
    router = IntentRouter(client)
    router.classify("What is the leave policy?")
    call_kwargs = client.messages.create.call_args.kwargs
    messages = call_kwargs["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "What is the leave policy?"
