"""
Baseline D — Full system: Schema RAG + self-correction loop.

SQLAgent retrieves relevant tables via SchemaRAG, generates SQL with the
selected Claude model, executes it, and retries up to 3 times feeding errors
back into the prompt. This is the thesis contribution — expected to be the
best config.

Usage
-----
  python -m baselines.baseline_d               # Claude, all 500 questions
  python -m baselines.baseline_d --limit 20    # Claude, quick smoke test
  python -m baselines.baseline_d --llm haiku   # Claude Haiku variant
"""
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from tqdm import tqdm

from src.utils import load_bird_questions
from src.config import BIRD_DEV_JSON, BIRD_DB_ROOT, LLM_OPTIONS
from src.llm_client import make_client
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

# Map --llm shorthand → LLM_OPTIONS key
_LLM_ALIASES = {
    "claude":      "Claude Sonnet 4.5",
    "haiku":       "Claude Haiku 4.5",
}


def run(
    llm_key: str = "Claude Sonnet 4.5",
    dev_json: str = BIRD_DEV_JSON,
    db_root: str = BIRD_DB_ROOT,
    limit: int = 0,
) -> Dict[str, float]:
    """
    Run Baseline D over the BIRD Mini-Dev set.

    Parameters
    ----------
    llm_key : str
        Key into LLM_OPTIONS (e.g. "Claude Sonnet 4.5").
    limit : int
        Cap number of questions (0 = all).

    Returns
    -------
    dict with execution_accuracy, total, correct.
    """
    if llm_key not in LLM_OPTIONS:
        raise ValueError(f"Unknown LLM key {llm_key!r}. Choose from: {list(LLM_OPTIONS)}")

    llm_cfg = LLM_OPTIONS[llm_key]
    logger.info("Baseline D — LLM: %s (%s)", llm_key, llm_cfg["model"])

    client = make_client(llm_cfg)
    rag    = SchemaRAG()
    agent  = SQLAgent(schema_rag=rag, client=client, model=llm_cfg["model"])

    questions = load_bird_questions(dev_json)
    if limit:
        questions = questions[:limit]

    slug = llm_key.lower().replace(" ", "_").replace(".", "").replace("/", "_")
    results_path = f"results/baseline_d_{slug}.json"

    indexed_dbs: set = set()
    results: List[Dict[str, Any]] = []
    correct = 0

    for item in tqdm(questions, desc=f"Baseline D [{llm_key}]"):
        db_id    = item["db_id"]
        question = item["question"]
        gold_sql = item["SQL"]
        evidence = item.get("evidence") or ""
        db_path  = db_path_for(db_id, db_root)

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
            "question_id":   item.get("question_id"),
            "db_id":         db_id,
            "question":      question,
            "gold_sql":      gold_sql,
            "pred_sql":      agent_result.sql,
            "match":         match,
            "attempts":      agent_result.attempts,
            "agent_success": agent_result.success,
            "error":         error or agent_result.error,
        })

    total = len(results)
    ex    = correct / total if total else 0.0

    save_results(results, results_path)
    log_to_mlflow(
        run_name=f"baseline_d_{slug}",
        params={
            "llm":             llm_key,
            "model":           llm_cfg["model"],
            "provider":        llm_cfg["provider"],
            "rag":             True,
            "self_correction": True,
            "limit":           limit or total,
        },
        metrics={
            "execution_accuracy": ex,
            "correct":            correct,
            "total":              total,
        },
        results_path=results_path,
    )

    logger.info("Baseline D [%s] — EX: %.2f%% (%d/%d)", llm_key, ex * 100, correct, total)
    return {"execution_accuracy": ex, "total": total, "correct": correct, "llm": llm_key}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Baseline D (RAG + self-correction)")
    parser.add_argument(
        "--llm",
        default="claude",
        choices=list(_LLM_ALIASES),
        help="Claude model for SQL generation (default: claude = Claude Sonnet 4.5).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max questions to evaluate (0 = all, default: 0)",
    )
    args = parser.parse_args()

    llm_key = _LLM_ALIASES[args.llm]
    metrics = run(llm_key=llm_key, limit=args.limit)
    print(
        f"\nBaseline D [{llm_key}]  EX = {metrics['execution_accuracy']*100:.1f}%"
        f"  ({metrics['correct']}/{metrics['total']})"
    )
