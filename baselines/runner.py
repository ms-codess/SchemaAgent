"""
Shared utilities for all baselines:
  - full schema serialization (no RAG)
  - execution accuracy (EX) comparison
  - MLflow run logging
  - result file writer
"""
import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow

from src.utils import get_db_connection, execute_sql, extract_sql_from_response
from src.config import BIRD_DB_ROOT, BIRD_DEV_JSON

logger = logging.getLogger(__name__)


# ── Schema helpers ────────────────────────────────────────────────────────────

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


def db_path_for(db_id: str, db_root: str = BIRD_DB_ROOT) -> str:
    return str(Path(db_root) / db_id / f"{db_id}.sqlite")


# ── Execution accuracy ────────────────────────────────────────────────────────

def _normalise(rows: Optional[List[Tuple]]) -> Optional[frozenset]:
    if rows is None:
        return None
    return frozenset(tuple(str(v) for v in row) for row in rows)


def execution_match(
    pred_sql: str,
    gold_sql: str,
    db_path: str,
) -> Tuple[bool, Optional[str]]:
    """
    Return (match: bool, error: str | None).
    match=True  → predicted result set equals gold result set.
    error       → execution error message if pred_sql failed, else None.
    """
    conn = get_db_connection(db_path)
    if conn is None:
        return False, "Could not open database"

    pred_rows, pred_err = execute_sql(conn, pred_sql)
    gold_rows, gold_err = execute_sql(conn, gold_sql)
    conn.close()

    if pred_err:
        return False, pred_err
    if gold_err:
        logger.warning("Gold SQL error (question skipped): %s", gold_err)
        return False, f"gold_error: {gold_err}"

    return _normalise(pred_rows) == _normalise(gold_rows), None


# ── Claude prompt builder ─────────────────────────────────────────────────────

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


def call_claude(client, question: str, schema_context: str, evidence: str = "") -> str:
    """Single-turn Claude call. Returns extracted SQL string."""
    msg = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_message(question, schema_context, evidence)}],
    )
    return extract_sql_from_response(msg.content[0].text)


def call_claude_with_correction(
    client, question: str, schema_context: str, evidence: str = "", max_attempts: int = 3
) -> Tuple[str, int]:
    """
    Multi-turn self-correction loop.
    Returns (final_sql, attempts_used).
    """
    messages = [{"role": "user", "content": build_user_message(question, schema_context, evidence)}]

    for attempt in range(1, max_attempts + 1):
        msg = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        reply = msg.content[0].text
        sql = extract_sql_from_response(reply)

        if attempt == max_attempts:
            return sql, attempt

        messages.append({"role": "assistant", "content": reply})
        return sql, attempt  # baseline_c caller handles the retry loop

    return "", max_attempts


# ── Results I/O ───────────────────────────────────────────────────────────────

def save_results(results: List[Dict[str, Any]], out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to %s", out_path)


# ── MLflow logging ────────────────────────────────────────────────────────────

def log_to_mlflow(
    run_name: str,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    results_path: str,
) -> None:
    mlflow.set_experiment("SchemaAgent-Ablation")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(results_path)
    logger.info("MLflow run '%s' logged. EX=%.4f", run_name, metrics.get("execution_accuracy", 0))
