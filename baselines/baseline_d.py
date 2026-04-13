"""
Baseline D — Full system: Schema RAG + self-correction loop.

SQLAgent retrieves relevant tables via SchemaRAG, generates SQL with Claude,
executes it, and retries up to 3 times feeding errors back into the prompt.
This is the thesis contribution — expected to be the best of all four configs.
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
from src.schema import SchemaRAG
from src.agent import SQLAgent
from baselines.runner import (
    db_path_for,
    execution_match,
    save_results,
    log_to_mlflow,
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_PATH = "results/baseline_d.json"


def run(
    dev_json: str = BIRD_DEV_JSON,
    db_root: str = BIRD_DB_ROOT,
    limit: int = 0,
) -> Dict[str, float]:
    """
    Run Baseline D over the BIRD Mini-Dev set.
    limit=0 means run all questions. Set limit=N for a quick smoke test.
    Returns {"execution_accuracy": float, "total": int, "correct": int}.
    """
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    rag = SchemaRAG()
    agent = SQLAgent(schema_rag=rag, client=client)

    questions = load_bird_questions(dev_json)
    if limit:
        questions = questions[:limit]

    indexed_dbs: set = set()
    results: List[Dict[str, Any]] = []
    correct = 0

    for item in tqdm(questions, desc="Baseline D"):
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

        results.append({
            "question_id": item.get("question_id"),
            "db_id": db_id,
            "question": question,
            "gold_sql": gold_sql,
            "pred_sql": agent_result.sql,
            "match": match,
            "attempts": agent_result.attempts,
            "agent_success": agent_result.success,
            "error": error or agent_result.error,
        })

    total = len(results)
    ex = correct / total if total else 0.0

    save_results(results, RESULTS_PATH)
    log_to_mlflow(
        run_name="baseline_d",
        params={"model": "claude-sonnet-4-5", "rag": True, "self_correction": True, "limit": limit or total},
        metrics={"execution_accuracy": ex, "correct": correct, "total": total},
        results_path=RESULTS_PATH,
    )

    logger.info("Baseline D — EX: %.2f%% (%d/%d)", ex * 100, correct, total)
    return {"execution_accuracy": ex, "total": total, "correct": correct}


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    run(limit=limit)
