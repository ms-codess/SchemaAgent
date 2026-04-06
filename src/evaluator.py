"""
Evaluator — feature/evaluation.

Runs all four ablation configurations (A, B, C, D) on the BIRD Mini-Dev set
and produces the thesis ablation table.

Configurations:
  A — no RAG, no self-correction   (floor)
  B — RAG only                     (proves RAG helps)
  C — self-correction only         (proves correction helps)
  D — full system: RAG + correction (target: beat MAC-SQL ~57-60%)

Each config runs the same 500 BIRD questions. For each question the generated
SQL is executed against the gold SQLite database and compared to the gold
result set (Execution Accuracy / EX).

All runs are logged to MLflow under the experiment "SchemaAgent-Ablation".
A summary JSON and a markdown ablation table are written to results/.

Usage:
    # Run all four configs on full BIRD Mini-Dev
    python -m src.evaluator

    # Smoke test — 10 questions only
    python -m src.evaluator --limit 10

    # Run a single config
    python -m src.evaluator --config a --limit 20
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import anthropic
from dotenv import load_dotenv
from tqdm import tqdm

import mlflow

from src.agent import SQLAgent
from src.config import BIRD_DEV_JSON, BIRD_DB_ROOT
from src.schema import SchemaRAG
from src.utils import load_bird_questions
from baselines.runner import (
    get_full_schema,
    db_path_for,
    execution_match,
    call_claude,
    save_results,
    log_to_mlflow,
    SYSTEM_PROMPT,
    build_user_message,
)
from src.utils import get_db_connection, execute_sql, extract_sql_from_response

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
MAX_ATTEMPTS = 3


# ── Config D runner (full system: RAG + self-correction via SQLAgent) ─────────

def run_config_d(
    questions: list,
    client: anthropic.Anthropic,
    db_root: str,
    limit: int = 0,
) -> Dict[str, Any]:
    """
    Config D — full system: Schema RAG + self-correction loop.
    Uses the production SQLAgent (src/agent.py).
    """
    rag = SchemaRAG()
    agent = SQLAgent(schema_rag=rag, client=client)

    if limit:
        questions = questions[:limit]

    results: List[Dict[str, Any]] = []
    correct = 0
    total_attempts = 0
    indexed_dbs: set = set()

    for item in tqdm(questions, desc="Config D (full system)"):
        db_id = item["db_id"]
        question = item["question"]
        gold_sql = item["SQL"]
        evidence = item.get("evidence") or ""
        db_path = db_path_for(db_id, db_root)

        if db_id not in indexed_dbs:
            if not rag.is_indexed(db_id):
                logger.info("Indexing %s …", db_id)
                rag.index(db_path, db_id)
            indexed_dbs.add(db_id)

        agent_result = agent.run(
            question=question,
            db_id=db_id,
            db_path=db_path,
            evidence=evidence,
        )

        match, error = execution_match(agent_result.sql, gold_sql, db_path)
        if match:
            correct += 1
        total_attempts += agent_result.attempts

        results.append({
            "question_id": item.get("question_id"),
            "db_id": db_id,
            "question": question,
            "gold_sql": gold_sql,
            "pred_sql": agent_result.sql,
            "match": match,
            "error": error or agent_result.error,
            "attempts": agent_result.attempts,
            "sql_success": agent_result.success,
        })

    total = len(results)
    ex = correct / total if total else 0.0
    avg_attempts = total_attempts / total if total else 0.0

    out_path = str(RESULTS_DIR / "config_d.json")
    save_results(results, out_path)
    log_to_mlflow(
        run_name="config_d_full_system",
        params={
            "model": "claude-sonnet-4-5",
            "rag": True,
            "self_correction": True,
            "max_attempts": MAX_ATTEMPTS,
            "limit": limit or total,
        },
        metrics={
            "execution_accuracy": ex,
            "correct": correct,
            "total": total,
            "avg_attempts": avg_attempts,
        },
        results_path=out_path,
    )
    logger.info("Config D — EX: %.2f%% (%d/%d) | avg attempts: %.2f", ex * 100, correct, total, avg_attempts)
    return {"config": "D", "rag": True, "self_correction": True,
            "execution_accuracy": ex, "correct": correct, "total": total, "avg_attempts": avg_attempts}


# ── Config A runner (no RAG, no correction) ───────────────────────────────────

def run_config_a(
    questions: list,
    client: anthropic.Anthropic,
    db_root: str,
    limit: int = 0,
) -> Dict[str, Any]:
    if limit:
        questions = questions[:limit]

    results, correct = [], 0
    for item in tqdm(questions, desc="Config A (no RAG, no correction)"):
        db_id = item["db_id"]
        question = item["question"]
        gold_sql = item["SQL"]
        evidence = item.get("evidence") or ""
        db_path = db_path_for(db_id, db_root)

        schema_context = get_full_schema(db_path)
        pred_sql = call_claude(client, question, schema_context, evidence)
        match, error = execution_match(pred_sql, gold_sql, db_path)
        if match:
            correct += 1

        results.append({
            "question_id": item.get("question_id"),
            "db_id": db_id, "question": question,
            "gold_sql": gold_sql, "pred_sql": pred_sql,
            "match": match, "error": error,
        })

    total = len(results)
    ex = correct / total if total else 0.0
    out_path = str(RESULTS_DIR / "config_a.json")
    save_results(results, out_path)
    log_to_mlflow(
        run_name="config_a_baseline",
        params={"model": "claude-sonnet-4-5", "rag": False, "self_correction": False, "limit": limit or total},
        metrics={"execution_accuracy": ex, "correct": correct, "total": total},
        results_path=out_path,
    )
    logger.info("Config A — EX: %.2f%% (%d/%d)", ex * 100, correct, total)
    return {"config": "A", "rag": False, "self_correction": False,
            "execution_accuracy": ex, "correct": correct, "total": total}


# ── Config B runner (RAG only) ────────────────────────────────────────────────

def run_config_b(
    questions: list,
    client: anthropic.Anthropic,
    db_root: str,
    limit: int = 0,
) -> Dict[str, Any]:
    rag = SchemaRAG()
    if limit:
        questions = questions[:limit]

    results, correct = [], 0
    indexed_dbs: set = set()

    for item in tqdm(questions, desc="Config B (RAG only)"):
        db_id = item["db_id"]
        question = item["question"]
        gold_sql = item["SQL"]
        evidence = item.get("evidence") or ""
        db_path = db_path_for(db_id, db_root)

        if db_id not in indexed_dbs:
            if not rag.is_indexed(db_id):
                rag.index(db_path, db_id)
            indexed_dbs.add(db_id)

        schema_context = rag.get_schema_context(question, db_id, use_rag=True)
        if not schema_context.strip():
            schema_context = get_full_schema(db_path)

        pred_sql = call_claude(client, question, schema_context, evidence)
        match, error = execution_match(pred_sql, gold_sql, db_path)
        if match:
            correct += 1

        results.append({
            "question_id": item.get("question_id"),
            "db_id": db_id, "question": question,
            "gold_sql": gold_sql, "pred_sql": pred_sql,
            "match": match, "error": error,
        })

    total = len(results)
    ex = correct / total if total else 0.0
    out_path = str(RESULTS_DIR / "config_b.json")
    save_results(results, out_path)
    log_to_mlflow(
        run_name="config_b_rag_only",
        params={"model": "claude-sonnet-4-5", "rag": True, "self_correction": False, "limit": limit or total},
        metrics={"execution_accuracy": ex, "correct": correct, "total": total},
        results_path=out_path,
    )
    logger.info("Config B — EX: %.2f%% (%d/%d)", ex * 100, correct, total)
    return {"config": "B", "rag": True, "self_correction": False,
            "execution_accuracy": ex, "correct": correct, "total": total}


# ── Config C runner (self-correction only) ────────────────────────────────────

def run_config_c(
    questions: list,
    client: anthropic.Anthropic,
    db_root: str,
    limit: int = 0,
) -> Dict[str, Any]:
    if limit:
        questions = questions[:limit]

    results, correct, total_attempts = [], 0, 0

    for item in tqdm(questions, desc="Config C (correction only)"):
        db_id = item["db_id"]
        question = item["question"]
        gold_sql = item["SQL"]
        evidence = item.get("evidence") or ""
        db_path = db_path_for(db_id, db_root)

        schema_context = get_full_schema(db_path)
        pred_sql, attempts = _self_correct(client, question, schema_context, db_path, evidence)
        match, error = execution_match(pred_sql, gold_sql, db_path)
        if match:
            correct += 1
        total_attempts += attempts

        results.append({
            "question_id": item.get("question_id"),
            "db_id": db_id, "question": question,
            "gold_sql": gold_sql, "pred_sql": pred_sql,
            "match": match, "error": error, "attempts": attempts,
        })

    total = len(results)
    ex = correct / total if total else 0.0
    avg_attempts = total_attempts / total if total else 0.0
    out_path = str(RESULTS_DIR / "config_c.json")
    save_results(results, out_path)
    log_to_mlflow(
        run_name="config_c_correction_only",
        params={"model": "claude-sonnet-4-5", "rag": False, "self_correction": True,
                "max_attempts": MAX_ATTEMPTS, "limit": limit or total},
        metrics={"execution_accuracy": ex, "correct": correct, "total": total, "avg_attempts": avg_attempts},
        results_path=out_path,
    )
    logger.info("Config C — EX: %.2f%% (%d/%d) | avg attempts: %.2f", ex * 100, correct, total, avg_attempts)
    return {"config": "C", "rag": False, "self_correction": True,
            "execution_accuracy": ex, "correct": correct, "total": total, "avg_attempts": avg_attempts}


def _self_correct(client, question, schema_context, db_path, evidence):
    """Inline self-correction loop for config C (mirrors baseline_c logic)."""
    messages = [{"role": "user", "content": build_user_message(question, schema_context, evidence)}]
    conn = get_db_connection(db_path)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        msg = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        reply = msg.content[0].text
        sql = extract_sql_from_response(reply)

        if attempt == MAX_ATTEMPTS:
            if conn:
                conn.close()
            return sql, attempt

        if conn:
            rows, err = execute_sql(conn, sql)
            if err is None:
                conn.close()
                return sql, attempt
            messages.append({"role": "assistant", "content": reply})
            messages.append({
                "role": "user",
                "content": (
                    f"The query produced this SQLite error:\n{err}\n\n"
                    "Fix it and return only the corrected SQL in ```sql ... ``` tags."
                ),
            })
        else:
            return sql, attempt

    if conn:
        conn.close()
    return sql, MAX_ATTEMPTS


# ── Summary table ─────────────────────────────────────────────────────────────

def build_summary(config_results: List[Dict[str, Any]]) -> str:
    """Render a markdown ablation table from the list of config result dicts."""
    lines = [
        "# SchemaAgent — Ablation Study Results",
        f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_",
        "",
        "| Config | RAG | Self-Correction | EX% | Correct | Total |",
        "|--------|-----|-----------------|-----|---------|-------|",
    ]
    for r in config_results:
        ex_pct = f"{r['execution_accuracy'] * 100:.2f}%"
        rag = "Yes" if r["rag"] else "No"
        sc = "Yes" if r["self_correction"] else "No"
        lines.append(
            f"| {r['config']} | {rag} | {sc} | {ex_pct} | {r['correct']} | {r['total']} |"
        )
    lines += [
        "",
        "## Target",
        "Beat MAC-SQL prompt-only (~57–60% EX). Floor: zero-shot Claude (~46%).",
    ]
    return "\n".join(lines)


# ── Main entry point ──────────────────────────────────────────────────────────

CONFIG_RUNNERS = {
    "a": run_config_a,
    "b": run_config_b,
    "c": run_config_c,
    "d": run_config_d,
}


def main():
    parser = argparse.ArgumentParser(description="Run SchemaAgent ablation evaluation on BIRD Mini-Dev.")
    parser.add_argument("--limit", type=int, default=0, help="Number of questions to run (0 = all 500).")
    parser.add_argument("--config", choices=["a", "b", "c", "d", "all"], default="all",
                        help="Which config to run (default: all).")
    parser.add_argument("--dev-json", default=BIRD_DEV_JSON)
    parser.add_argument("--db-root", default=BIRD_DB_ROOT)
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    questions = load_bird_questions(args.dev_json)
    logger.info("Loaded %d questions from %s", len(questions), args.dev_json)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    configs_to_run = list(CONFIG_RUNNERS.keys()) if args.config == "all" else [args.config]
    all_results = []

    for cfg in configs_to_run:
        logger.info("=== Running config %s ===", cfg.upper())
        result = CONFIG_RUNNERS[cfg](questions, client, args.db_root, limit=args.limit)
        all_results.append(result)

    # Write summary JSON
    summary_json_path = RESULTS_DIR / "ablation_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Summary written to %s", summary_json_path)

    # Write markdown table
    if len(all_results) > 1:
        md_table = build_summary(all_results)
        md_path = RESULTS_DIR / "ablation_table.md"
        md_path.write_text(md_table, encoding="utf-8")
        logger.info("Ablation table written to %s", md_path)
        print("\n" + md_table)


if __name__ == "__main__":
    main()
