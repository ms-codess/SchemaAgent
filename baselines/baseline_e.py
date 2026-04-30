"""
Baseline E — Schema RAG + DocRAG on BIRD Data Dictionaries + Self-correction.

Extends Config D by replacing the gold evidence hint with retrieved domain
knowledge from BIRD's database_description/*.csv files (one per table per DB).
These CSVs describe every column — its human-readable name, meaning, data
format, and valid values — acting as an enterprise-style data dictionary.

Two conditions
--------------
  --mode no_doc    (E-noDoc)
      Same as Config D but the gold evidence hint is dropped entirely.
      Quantifies how much Config D's accuracy depended on the hint.

  --mode with_doc  (E-withDoc)
      Same as E-noDoc but retrieves column descriptions from the BIRD CSVs
      via BirdDescRAG and passes them as the hint instead of the gold string.
      Shows how much of the hint's value is recoverable from a data dictionary.

Three-way thesis comparison
---------------------------
  Config D     (gold hint)           63.2%  — EX with perfect domain knowledge
  Config E-noDoc (no hint)              ?%  — EX without any domain knowledge
  Config E-withDoc (DocRAG dict)        ?%  — EX with retrieved domain knowledge
                                              recovery = E-withDoc - E-noDoc

Usage
-----
  python -m baselines.baseline_e --mode no_doc
  python -m baselines.baseline_e --mode with_doc
  python -m baselines.baseline_e --mode no_doc --limit 20    # smoke test
  python -m baselines.baseline_e --mode with_doc --limit 20
"""

import argparse
import logging
import os
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from tqdm import tqdm

from src.utils import load_bird_questions
from src.config import BIRD_DEV_JSON, BIRD_DB_ROOT, LLM_OPTIONS
from src.llm_client import make_client
from src.schema import SchemaRAG
from src.agent import SQLAgent
from src.bird_desc_rag import BirdDescRAG
from baselines.runner import (
    db_path_for,
    execution_match,
    save_results,
    log_to_mlflow,
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_DEFAULT_LLM = "Claude Sonnet 4.5"


def run(
    mode: str,
    llm_key: str = _DEFAULT_LLM,
    dev_json: str = BIRD_DEV_JSON,
    db_root: str = BIRD_DB_ROOT,
    limit: int = 0,
) -> Dict[str, Any]:
    """
    Run Baseline E (no_doc or with_doc mode) over the BIRD Mini-Dev set.

    Parameters
    ----------
    mode : str
        "no_doc"   — drop the gold evidence, no retrieval.
        "with_doc" — replace gold evidence with BirdDescRAG retrieval.
    llm_key : str
        Key into LLM_OPTIONS for the Claude model used for SQL generation.
    limit : int
        Cap number of questions (0 = all).

    Returns
    -------
    dict with execution_accuracy, correct, total, mode.
    """
    if mode not in ("no_doc", "with_doc"):
        raise ValueError(f"mode must be 'no_doc' or 'with_doc', got {mode!r}")
    if llm_key not in LLM_OPTIONS:
        raise ValueError(f"Unknown LLM key {llm_key!r}. Choose from: {list(LLM_OPTIONS)}")

    llm_cfg  = LLM_OPTIONS[llm_key]
    run_name = f"baseline_e_{mode}"
    logger.info("Baseline E [%s] — LLM: %s  mode: %s", run_name, llm_key, mode)

    client = make_client(llm_cfg)
    rag    = SchemaRAG()
    agent  = SQLAgent(schema_rag=rag, client=client, model=llm_cfg["model"])
    desc_rag = BirdDescRAG() if mode == "with_doc" else None

    questions = load_bird_questions(dev_json)
    if limit:
        questions = questions[:limit]

    results_path = f"results/{run_name}.json"
    indexed_dbs: set = set()
    results: List[Dict[str, Any]] = []
    correct = 0

    for item in tqdm(questions, desc=f"Baseline E [{mode}]"):
        db_id    = item["db_id"]
        question = item["question"]
        gold_sql = item["SQL"]
        db_path  = db_path_for(db_id, db_root)

        # Index schema (same as Config D)
        if db_id not in indexed_dbs:
            if not rag.is_indexed(db_id):
                logger.info("Indexing schema for %s …", db_id)
                rag.index(db_path, db_id)
            # Index description CSVs if in with_doc mode
            if desc_rag is not None and not desc_rag.is_indexed(db_id):
                logger.info("Indexing data dictionary for %s …", db_id)
                desc_rag.index_db(db_root, db_id)
            indexed_dbs.add(db_id)

        # Build the evidence string
        if mode == "no_doc":
            evidence = ""
        else:
            evidence = desc_rag.retrieve(question, db_id=db_id)

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
            "mode":          mode,
            "evidence_used": bool(evidence),
        })

    total = len(results)
    ex    = correct / total if total else 0.0

    save_results(results, results_path)
    log_to_mlflow(
        run_name=run_name,
        params={
            "mode":            mode,
            "llm":             llm_key,
            "model":           llm_cfg["model"],
            "rag":             True,
            "self_correction": True,
            "doc_rag":         mode == "with_doc",
            "gold_evidence":   False,
            "limit":           limit or total,
        },
        metrics={
            "execution_accuracy": ex,
            "correct":            correct,
            "total":              total,
        },
        results_path=results_path,
    )

    logger.info(
        "Baseline E [%s] — EX: %.2f%%  (%d/%d)", mode, ex * 100, correct, total
    )
    return {
        "execution_accuracy": ex,
        "total": total,
        "correct": correct,
        "mode": mode,
        "llm": llm_key,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Baseline E — Schema RAG + BirdDescRAG + self-correction (no gold hint)"
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["no_doc", "with_doc"],
        help=(
            "no_doc   - drop evidence entirely (ablation)\n"
            "with_doc - replace evidence with DocRAG over BIRD description CSVs"
        ),
    )
    parser.add_argument(
        "--llm",
        default="claude",
        choices=["claude", "haiku"],
        help="Claude model for SQL generation (default: claude = Claude Sonnet 4.5)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max questions to evaluate (0 = all, default: 0)",
    )
    args = parser.parse_args()

    _LLM_ALIASES = {"claude": "Claude Sonnet 4.5", "haiku": "Claude Haiku 4.5"}
    llm_key = _LLM_ALIASES[args.llm]
    metrics = run(mode=args.mode, llm_key=llm_key, limit=args.limit)
    print(
        f"\nBaseline E [{args.mode}] [{llm_key}]"
        f"  EX = {metrics['execution_accuracy']*100:.1f}%"
        f"  ({metrics['correct']}/{metrics['total']})"
    )
