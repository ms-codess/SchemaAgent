# SchemaAgent Codebase Guide

## What this project is

SchemaAgent is a thesis research project: a hybrid Text-to-SQL system evaluated on BIRD Mini-Dev (500 questions, 11 SQLite databases). The central result is that Schema RAG and self-correction are super-additive: Config D reaches 63.2% execution accuracy versus the 60.2% zero-shot baseline.

The current thesis extension is Config E:

- `E-noDoc` removes BIRD's gold evidence hint
- `E-withDoc` replaces that hint with BirdDescRAG retrieval over `database_description/*.csv`

This tests whether enterprise-style data dictionaries can recover missing domain knowledge.

## Architecture in one paragraph

`HybridFusion` (`src/fusion.py`) is the top-level orchestrator. It calls `IntentRouter` (`src/router/`) to classify the question as database, document, or hybrid, then runs `SQLAgent` (`src/agent.py`) for SQL and/or `DocRAG` (`src/doc_rag.py`) for document passages, then synthesises a natural-language answer with Claude. `SchemaRAG` (`src/schema/`) provides semantic table retrieval via ChromaDB embeddings, and `BirdDescRAG` (`src/bird_desc_rag.py`) provides BIRD data-dictionary retrieval for Config E.

## Key design decisions

- **SQL generation is pluggable:** `SQLAgent` and `HybridFusion` accept any `UnifiedClient` (`src/llm_client.py`). Routing and synthesis always use Claude for reliability.
- **Dual client:** `HybridFusion` has `client` for SQL generation and `synth_client` for answer synthesis.
- **Schema escalation:** If RAG context caused an execution error, `SQLAgent` automatically falls back to the full schema on the next attempt.
- **Prompt caching:** Anthropic calls in `agent.py` and `fusion.py` use `cache_control` blocks. `UnifiedClient` strips these for non-Anthropic backends.

## LLM model registry

All model names and providers live in `src/config.py -> LLM_OPTIONS`. Adding a new model should only require one dict entry there.

## BIRD data paths

BIRD Mini-Dev data lives in `_bird_data/minidev/MINIDEV/`.

- JSON: `_bird_data/minidev/MINIDEV/mini_dev_sqlite.json`
- Databases: `_bird_data/minidev/MINIDEV/dev_databases/<db_id>/<db_id>.sqlite`
- Data dictionaries: `_bird_data/minidev/MINIDEV/dev_databases/<db_id>/database_description/*.csv`

Paths are configured in `src/config.py` and can be overridden with `BIRD_DB_ROOT` and `BIRD_DEV_JSON`.

## Running evaluation

```bash
python -m baselines.baseline_d --llm claude
python -m baselines.baseline_d --llm claude --limit 20
python -m baselines.baseline_d --llm qwen
python -m baselines.baseline_e --mode no_doc
python -m baselines.baseline_e --mode with_doc
```

Results go to `results/` and MLflow experiment `SchemaAgent-Ablation`.

## Running the UI

```bash
streamlit run app.py
```

The LLM selector lives in the sidebar Settings expander.

## Environment variables needed

| Var | Required for |
|-----|--------------|
| `ANTHROPIC_API_KEY` | All modes, including routing, synthesis, and Claude SQL generation |
| `OPENROUTER_API_KEY` | Qwen via OpenRouter |

## Test suite

```bash
pytest tests/ -v
```

## File map

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI |
| `src/agent.py` | SQLAgent with three-attempt correction loop |
| `src/fusion.py` | HybridFusion orchestrator |
| `src/router/__init__.py` | IntentRouter |
| `src/schema/` | SchemaRAG serializer, indexer, retriever |
| `src/doc_rag.py` | DocRAG for uploaded documents |
| `src/bird_desc_rag.py` | BirdDescRAG for BIRD data dictionaries |
| `src/llm_client.py` | Unified client adapter |
| `src/config.py` | Model names, paths, and LLM registry |
| `src/utils.py` | BIRD loader, DB connection, SQL extractor |
| `baselines/runner.py` | Shared schema dump, EX matching, MLflow logging |
| `baselines/baseline_{a-e}.py` | Ablation configurations |
| `results/` | Evaluation outputs and logs |
