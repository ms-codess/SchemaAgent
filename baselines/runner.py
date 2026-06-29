"""
Shared utilities for all baselines:
  - full schema serialization (no RAG)
  - execution accuracy (EX) comparison
  - MLflow run logging
  - result file writer
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow

from src.utils import (
    get_db_connection,
    execute_sql,
    extract_sql_from_response,
    SYSTEM_PROMPT,
    build_user_message,
    get_full_schema,
)
from src.config import BIRD_DB_ROOT

logger = logging.getLogger(__name__)


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


# ── Claude call ───────────────────────────────────────────────────────────────

def call_claude(client, question: str, schema_context: str, evidence: str = "") -> str:
    """Single-turn Claude call. Returns extracted SQL string."""
    msg = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_message(question, schema_context, evidence)}],
    )
    return extract_sql_from_response(msg.content[0].text)


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
