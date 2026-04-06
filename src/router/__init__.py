"""
Intent Router — feature/intent-router.

Classifies a natural language question into one of three retrieval paths:
  database  — answer lives entirely in the DB, needs SQL
  document  — answer lives in policy/doc passages, no DB needed
  hybrid    — needs both a DB result AND a doc passage

One Claude Haiku call per question. Returns a RouteResult dataclass with
intent, confidence, and a one-line reasoning string useful for debugging
and thesis error analysis.

Fallback rule: any malformed response, unexpected label, or API error
defaults to intent="hybrid", confidence="low". Hybrid is the safe path —
it runs both retrieval pipelines rather than silently skipping one.

Usage:
    import anthropic
    client = anthropic.Anthropic()
    router = IntentRouter(client)
    result = router.classify("How many students enrolled in 2023?")
    print(result.intent, result.confidence, result.reasoning)
"""

import json
import logging
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

VALID_INTENTS = {"database", "document", "hybrid"}
VALID_CONFIDENCES = {"high", "medium", "low"}
DEFAULT_INTENT: Literal["hybrid"] = "hybrid"
DEFAULT_CONFIDENCE: Literal["low"] = "low"

_SYSTEM_PROMPT = """\
You are an intent classifier for a hybrid question-answering system with access to two sources:
1. A relational DATABASE — answers questions about records, counts, aggregations, and structured data via SQL.
2. DOCUMENTS — policy files, reports, and unstructured text (PDFs, Word docs).

Classify the user's question into exactly one of these three intents:

DATABASE — The answer requires querying the database. The question asks about specific records,
counts, aggregations, comparisons, or any structured data.
Examples:
  - "How many students enrolled in the Computer Science program in 2023?"
  - "What is the average salary of employees in the engineering department?"

DOCUMENT — The answer is found in policy documents or unstructured text. No database query needed.
Examples:
  - "What is the academic probation GPA threshold?"
  - "What are the requirements to apply for a leave of absence?"

HYBRID — The answer requires BOTH a database result AND information from documents.
Examples:
  - "Does Alice's current GPA meet the graduation requirements?"
  - "Which employees are below the minimum wage set in the compensation policy?"

Respond with ONLY a JSON object — no markdown, no explanation:
{"intent": "database" | "document" | "hybrid", "confidence": "high" | "medium" | "low", "reasoning": "<one sentence>"}
"""


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class RouteResult:
    intent: Literal["database", "document", "hybrid"]
    confidence: Literal["high", "medium", "low"]
    reasoning: str


# ── Router ────────────────────────────────────────────────────────────────────

class IntentRouter:
    """
    Classify a natural language question into a retrieval intent.

    Parameters
    ----------
    client : anthropic.Anthropic
        Authenticated Anthropic API client.
    model : str
        Claude model to use. Defaults to Haiku — classification is simple
        pattern matching that does not need Sonnet-level reasoning, and
        Haiku is ~20x cheaper per call.
    """

    def __init__(self, client, model: str = "claude-haiku-4-5-20251001"):
        self.client = client
        self.model = model

    # ── Public interface ──────────────────────────────────────────────────────

    def classify(self, question: str) -> RouteResult:
        """
        Classify a question and return a RouteResult.

        Never raises — any failure (API error, malformed JSON, unknown label)
        defaults to intent="hybrid", confidence="low".
        """
        try:
            raw = self._call_claude(question)
            result = self._parse_response(raw)
        except Exception as exc:
            logger.warning("IntentRouter error (defaulting to hybrid): %s", exc)
            result = RouteResult(
                intent=DEFAULT_INTENT,
                confidence=DEFAULT_CONFIDENCE,
                reasoning="fallback due to error",
            )

        self._log_to_mlflow(result)
        logger.debug(
            "Classified %r → %s (%s): %s",
            question[:60], result.intent, result.confidence, result.reasoning,
        )
        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _call_claude(self, question: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=128,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": question}],
        )
        return response.content[0].text.strip()

    def _parse_response(self, raw: str) -> RouteResult:
        """
        Parse Claude's JSON response into a RouteResult.
        Returns a hybrid fallback on any parse failure.
        """
        text = raw.strip()

        # Strip accidental markdown fences (Claude occasionally wraps JSON)
        if text.startswith("```"):
            parts = text.split("```")
            # parts[1] is the content between the first pair of fences
            inner = parts[1] if len(parts) > 1 else ""
            if inner.lower().startswith("json"):
                inner = inner[4:]
            text = inner.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse router JSON: %s | raw=%r", exc, raw[:120])
            return RouteResult(
                intent=DEFAULT_INTENT,
                confidence=DEFAULT_CONFIDENCE,
                reasoning="malformed JSON response",
            )

        intent = str(data.get("intent", "")).lower().strip()
        confidence = str(data.get("confidence", "")).lower().strip()
        reasoning = str(data.get("reasoning", "")).strip()

        if intent not in VALID_INTENTS:
            logger.warning("Unexpected intent %r — defaulting to hybrid", intent)
            return RouteResult(
                intent=DEFAULT_INTENT,
                confidence=DEFAULT_CONFIDENCE,
                reasoning=f"unknown intent label: {intent!r}",
            )

        if confidence not in VALID_CONFIDENCES:
            logger.debug("Unexpected confidence %r — defaulting to low", confidence)
            confidence = DEFAULT_CONFIDENCE

        return RouteResult(
            intent=intent,  # type: ignore[arg-type]
            confidence=confidence,  # type: ignore[arg-type]
            reasoning=reasoning or "no reasoning provided",
        )

    @staticmethod
    def _log_to_mlflow(result: RouteResult) -> None:
        """Log intent and confidence into the active MLflow run, if one exists."""
        try:
            import mlflow
            if mlflow.active_run() is not None:
                mlflow.log_params({
                    "router_intent": result.intent,
                    "router_confidence": result.confidence,
                })
        except Exception:
            pass  # MLflow not installed or no active run — silently skip
