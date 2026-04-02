"""
Baseline A — Zero-shot, no RAG, no self-correction.

Full schema is injected into the prompt. Claude generates SQL in one shot.
This is the floor: expected ~46% EX on BIRD Mini-Dev.
"""
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

import anthropic
from dotenv import load_dotenv
from tqdm import tqdm

from src.utils import load_bird_questions
from src.config import BIRD_DEV_JSON, BIRD_DB_ROOT
from baselines.runner import (
    get_full_schema,
    db_path_for,
    execution_match,
    call_claude,
    save_results,
    log_to_mlflow,
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_PATH = "results/baseline_a.json"


def run(
    dev_json: str = BIRD_DEV_JSON,
    db_root: str = BIRD_DB_ROOT,
    limit: int = 0,
) -> Dict[str, float]:
    """
    Run Baseline A over the BIRD Mini-Dev set.
    limit=0 means run all questions. Set limit=N for a quick smoke test.
    Returns {"execution_accuracy": float, "total": int, "correct": int}.
    """
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    questions = load_bird_questions(dev_json)
    if limit:
        questions = questions[:limit]

    results: List[Dict[str, Any]] = []
    correct = 0

    for item in tqdm(questions, desc="Baseline A"):
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
            "db_id": db_id,
            "question": question,
            "gold_sql": gold_sql,
            "pred_sql": pred_sql,
            "match": match,
            "error": error,
        })

    total = len(results)
    ex = correct / total if total else 0.0

    save_results(results, RESULTS_PATH)
    log_to_mlflow(
        run_name="baseline_a",
        params={"model": "claude-sonnet-4-5", "rag": False, "self_correction": False, "limit": limit or total},
        metrics={"execution_accuracy": ex, "correct": correct, "total": total},
        results_path=RESULTS_PATH,
    )

    logger.info("Baseline A — EX: %.2f%% (%d/%d)", ex * 100, correct, total)
    return {"execution_accuracy": ex, "total": total, "correct": correct}


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    run(limit=limit)
