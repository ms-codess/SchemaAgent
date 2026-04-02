"""
Baseline C — Self-correction only, no RAG.

Full schema is injected (no retrieval). If the generated SQL fails execution,
the error is fed back to Claude for up to 3 attempts.
Expected ~49% EX on BIRD Mini-Dev — proves self-correction alone helps.
"""
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

import anthropic
from dotenv import load_dotenv
from tqdm import tqdm

from src.utils import load_bird_questions, get_db_connection, execute_sql, extract_sql_from_response
from src.config import BIRD_DEV_JSON, BIRD_DB_ROOT
from baselines.runner import (
    get_full_schema,
    db_path_for,
    execution_match,
    build_user_message,
    SYSTEM_PROMPT,
    save_results,
    log_to_mlflow,
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_PATH = "results/baseline_c.json"
MAX_ATTEMPTS = 3


def call_with_correction(
    client: anthropic.Anthropic,
    question: str,
    schema_context: str,
    db_path: str,
    evidence: str = "",
) -> Tuple[str, int]:
    """
    Self-correction loop: run SQL, feed errors back to Claude, retry up to MAX_ATTEMPTS.
    Returns (final_sql, attempts_used).
    """
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

        # On the last attempt return whatever we have
        if attempt == MAX_ATTEMPTS:
            if conn:
                conn.close()
            return sql, attempt

        # Try to execute — if it works, no need to retry
        if conn:
            rows, err = execute_sql(conn, sql)
            if err is None:
                conn.close()
                return sql, attempt

            # Append assistant reply + error feedback for next turn
            messages.append({"role": "assistant", "content": reply})
            messages.append({
                "role": "user",
                "content": (
                    f"The query you wrote produced this SQLite error:\n{err}\n\n"
                    "Please fix it and return only the corrected SQL wrapped in ```sql ... ``` tags."
                ),
            })
        else:
            # Can't validate — just return first attempt
            return sql, attempt

    if conn:
        conn.close()
    return sql, MAX_ATTEMPTS


def run(
    dev_json: str = BIRD_DEV_JSON,
    db_root: str = BIRD_DB_ROOT,
    limit: int = 0,
) -> Dict[str, float]:
    """
    Run Baseline C over the BIRD Mini-Dev set.
    limit=0 means run all questions. Set limit=N for a quick smoke test.
    Returns {"execution_accuracy": float, "total": int, "correct": int}.
    """
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    questions = load_bird_questions(dev_json)
    if limit:
        questions = questions[:limit]

    results: List[Dict[str, Any]] = []
    correct = 0
    total_attempts = 0

    for item in tqdm(questions, desc="Baseline C"):
        db_id = item["db_id"]
        question = item["question"]
        gold_sql = item["SQL"]
        evidence = item.get("evidence") or ""
        db_path = db_path_for(db_id, db_root)

        schema_context = get_full_schema(db_path)
        pred_sql, attempts = call_with_correction(client, question, schema_context, db_path, evidence)
        match, error = execution_match(pred_sql, gold_sql, db_path)

        if match:
            correct += 1
        total_attempts += attempts

        results.append({
            "question_id": item.get("question_id"),
            "db_id": db_id,
            "question": question,
            "gold_sql": gold_sql,
            "pred_sql": pred_sql,
            "match": match,
            "error": error,
            "attempts": attempts,
        })

    total = len(results)
    ex = correct / total if total else 0.0
    avg_attempts = total_attempts / total if total else 0.0

    save_results(results, RESULTS_PATH)
    log_to_mlflow(
        run_name="baseline_c",
        params={
            "model": "claude-sonnet-4-5",
            "rag": False,
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
        results_path=RESULTS_PATH,
    )

    logger.info(
        "Baseline C — EX: %.2f%% (%d/%d) | avg attempts: %.2f",
        ex * 100, correct, total, avg_attempts,
    )
    return {"execution_accuracy": ex, "total": total, "correct": correct, "avg_attempts": avg_attempts}


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    run(limit=limit)
