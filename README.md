# SchemaAgent

A hybrid Text-to-SQL system that answers natural language questions over SQLite databases. It combines **semantic schema retrieval (RAG)** with an **agentic self-correction loop**, and can optionally fuse database results with policy documents. Evaluated on the BIRD Mini-Dev benchmark (500 questions, 11 databases).

This is a thesis research project. The core result is that Schema RAG and self-correction are **super-additive**: combining them in Config D reaches **63.2% EX** over 500 BIRD Mini-Dev questions, above either component alone.

---

## Thesis Purpose

The thesis asks a practical systems question: how should a Text-to-SQL agent compensate for the fact that real enterprise databases are hard to query from schema alone?

The project studies three sources of improvement:

1. **Schema RAG** narrows the prompt to the most relevant tables instead of dumping the full schema.
2. **Self-correction** lets the agent repair syntax, execution, and empty-result failures over multiple attempts.
3. **Document RAG over data dictionaries** supplies missing business semantics when schemas do not encode value meanings, aliases, formulas, or date conventions.

The goal is not just to improve benchmark accuracy. It is to show that enterprise-grade Text-to-SQL needs both structured schema grounding and unstructured documentation grounding.

---

## Thesis Abstract

Text-to-SQL systems often fail in enterprise settings because SQL generation depends on information that is not fully expressed in the relational schema. Column names may be ambiguous, value encodings may be opaque, and important business rules may exist only in external documentation. This thesis presents SchemaAgent, a hybrid Text-to-SQL architecture that combines semantic schema retrieval, an agentic self-correction loop, and document retrieval over auxiliary data sources. On the BIRD Mini-Dev benchmark, the main system configuration achieves 63.2% execution accuracy, outperforming zero-shot prompting and single-component ablations. A second experiment replaces BIRD's gold evidence hints with retrieval over `database_description/*.csv` files, treating them as enterprise-style data dictionaries. This evaluates whether document retrieval can recover domain knowledge that would otherwise be unavailable at inference time. The thesis argues that robust Text-to-SQL requires a layered grounding strategy: schema retrieval identifies the structural search space, self-correction repairs execution-time failures, and document retrieval supplies the semantic context needed to bridge the gap between database structure and business meaning.

The same abstract is available in [docs/THESIS_ABSTRACT.md](docs/THESIS_ABSTRACT.md).

---

## Architecture

```text
User Question
    |
IntentRouter (Claude Haiku) -> "database" | "document" | "hybrid"
    |
HybridFusion.answer()
    |-- [database / hybrid] SQLAgent
    |       |-- SchemaRAG -> top-k relevant tables via ChromaDB embeddings
    |       |-- Claude: generate SQL
    |       +-- Self-correction loop (up to 3 attempts on syntax/exec/empty errors)
    |-- [document / hybrid] DocRAG -> relevant passages from uploaded PDFs/DOCX
    +-- Synthesis (Claude Haiku/Sonnet) -> natural language answer
```

**Key design choice:** the system is Claude-based end to end. You can switch between Claude Sonnet and Claude Haiku for SQL generation, while routing and synthesis remain on Claude.

---

## Quick Start

### 1. Clone and install

```bash
git clone <repo>
cd SchemaAgent
pip install -r requirements.txt
```

### 2. Set API keys in `.env`

```env
ANTHROPIC_API_KEY=sk-ant-...        # required for all modes
```

### 3. Download BIRD Mini-Dev data

```bash
# Download from https://bird-bench.github.io/
# Extract so the structure is:
# _bird_data/minidev/MINIDEV/mini_dev_sqlite.json
# _bird_data/minidev/MINIDEV/dev_databases/<db_id>/<db_id>.sqlite
```

### 4. Launch the UI

```bash
streamlit run app.py
```

Connect a SQLite database in the sidebar, choose your Claude model in Settings, and start asking questions.

---

## Evaluation

```bash
# Claude Sonnet 4.5 -- thesis contribution model
python -m baselines.baseline_d --llm claude

# Claude Haiku 4.5 variant
python -m baselines.baseline_d --llm haiku

# Quick smoke test
python -m baselines.baseline_d --llm claude --limit 20

# Config E -- no gold hint
python -m baselines.baseline_e --mode no_doc

# Config E -- BirdDescRAG over BIRD data dictionaries
python -m baselines.baseline_e --mode with_doc
```

Results are saved to `results/` and logged to MLflow.

As of April 30, 2026, fresh reruns are currently blocked by Anthropic billing: a new Config A run failed on question 1 with an insufficient-credit API error. The repo still contains the full April 25, 2026 Config D result, an interrupted April 25, 2026 `E-noDoc` run that stopped at 330/500 for the same reason, and 5-question smoke outputs for both Config E modes.

---

## Ablation Results (BIRD Mini-Dev, 500 questions)

| Config | LLM | Description | EX% | Correct | Delta |
|--------|-----|-------------|-----|---------|-------|
| A | Claude Sonnet 4.5 | Zero-shot (full schema, no aids) | 60.2% | 301/500 | -- |
| B | Claude Sonnet 4.5 | Schema RAG only | 61.2% | 306/500 | +1.0 pp |
| C | Claude Sonnet 4.5 | Self-correction only | 60.8% | 304/500 | +0.6 pp |
| **D** | **Claude Sonnet 4.5** | **RAG + self-correction + gold hint** | **63.2%** | **316/500** | **+3.0 pp** |
| E-noDoc | Claude Sonnet 4.5 | RAG + self-correction, no hint | pending | -- | -- |
| E-withDoc | Claude Sonnet 4.5 | RAG + self-correction + BirdDescRAG | pending | -- | -- |

**Key findings:** Configs A-D show that RAG and self-correction are super-additive. Config E extends the thesis by testing whether BirdDescRAG can recover domain knowledge from BIRD's `database_description/*.csv` files after the gold evidence hint is removed.

See [results/EVALUATION_RESULTS.md](results/EVALUATION_RESULTS.md) for the full breakdown.

---

## Project Structure

```text
SchemaAgent/
|-- app.py                    # Streamlit web UI
|-- src/
|   |-- agent.py              # SQLAgent -- generation + self-correction loop
|   |-- fusion.py             # HybridFusion -- routes and synthesises answers
|   |-- router/               # IntentRouter -- classifies database/document/hybrid
|   |-- schema/               # SchemaRAG -- serializer, indexer, retriever
|   |-- doc_rag.py            # DocRAG -- PDF/DOCX/TXT indexing and retrieval
|   |-- bird_desc_rag.py      # BirdDescRAG -- BIRD data-dictionary retrieval
|   |-- llm_client.py         # Anthropic client wrapper used by UI and baselines
|   +-- config.py             # Centralised Claude model names and data paths
|-- baselines/
|   |-- baseline_a.py         # Config A: zero-shot
|   |-- baseline_b.py         # Config B: RAG only
|   |-- baseline_c.py         # Config C: self-correction only
|   |-- baseline_d.py         # Config D: full system (--llm claude|haiku)
|   |-- baseline_e.py         # Config E: no hint vs BirdDescRAG replacement
|   +-- runner.py             # Shared: schema dump, execution match, MLflow logging
|-- tests/                    # Unit tests for all components
|-- data/
|   +-- uw_courses/           # Sample SQLite database (tracked in git)
|-- docs/
|   |-- ARCHITECTURE.md       # Detailed architecture and data flow
|   +-- THESIS_ABSTRACT.md    # Thesis-ready abstract
+-- results/                  # Evaluation outputs
```

---

## Claude Configuration

Models are registered in `src/config.py`.

| UI Label | Provider | Model | API Key |
|----------|----------|-------|---------|
| Claude Sonnet 4.5 | Anthropic | claude-sonnet-4-5 | `ANTHROPIC_API_KEY` |
| Claude Haiku 4.5 | Anthropic | claude-haiku-4-5-20251001 | `ANTHROPIC_API_KEY` |

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Technologies

- **Claude Sonnet/Haiku** -- SQL generation, intent routing, answer synthesis
- **ChromaDB** + `all-MiniLM-L6-v2` -- schema and document vector store
- **SQLite** + `sqlglot` -- database execution and SQL validation
- **Streamlit** -- web UI
- **MLflow** -- experiment tracking
- **BIRD Mini-Dev** -- evaluation benchmark
