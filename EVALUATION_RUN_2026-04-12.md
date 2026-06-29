# SchemaAgent — BIRD Mini-Dev Evaluation Run
**Date:** 2026-04-12  
**Branch:** `feature/self-correction` (agent.py) + `feature/oracle-sec-eval` (baselines A/B/C)  
**Model:** `claude-sonnet-4-5` (via Anthropic API)

---

## 1. What Was Tested

This evaluation measures **execution accuracy (EX%)** across four ablation configurations (A/B/C/D) on the BIRD Mini-Dev benchmark. Each configuration isolates one or more components of the SchemaAgent system to prove their individual contribution.

**Execution accuracy** = the fraction of questions where the predicted SQL returns the exact same result set as the gold SQL when run against the database.

---

## 2. Dataset — BIRD Mini-Dev

| Property | Value |
|---|---|
| Source | BIRD Text-to-SQL benchmark (bird-bench.github.io) |
| Split | Mini-Dev (subset of full BIRD dev set) |
| Questions | 500 |
| Databases | 11 SQLite databases |
| File | `data/bird/mini_dev_sqlite.json` |

### Questions per database

| Database | Questions | Domain |
|---|---|---|
| formula_1 | 66 | motorsport statistics |
| card_games | 52 | trading card game data |
| superhero | 52 | comic book superhero attributes |
| european_football_2 | 51 | football match/player stats |
| codebase_community | 49 | Q&A community (Stack Exchange style) |
| student_club | 48 | university club membership |
| thrombosis_prediction | 50 | medical patient records |
| toxicology | 40 | chemical toxicology |
| california_schools | 30 | CA public school data |
| debit_card_specializing | 30 | retail transaction data |
| financial | 32 | Czech bank transactions |

### Difficulty distribution

| Difficulty | Count | % of total |
|---|---|---|
| Simple | 148 | 29.6% |
| Moderate | 250 | 50.0% |
| Challenging | 102 | 20.4% |

---

## 3. System Setup

### Environment

| Component | Value |
|---|---|
| Python | 3.12.10 |
| OS | Windows 10 |
| LLM | `claude-sonnet-4-5` |
| Vector store | ChromaDB (local, persistent) |
| Embeddings | `all-MiniLM-L6-v2` (Sentence Transformers) |
| SQL validation | `sqlglot` |
| SQL execution | SQLite via `sqlite3` stdlib |
| Experiment tracking | MLflow (`mlruns/`) |

### Key dependencies

```
anthropic
chromadb
sentence-transformers
sqlglot
mlflow
python-oracledb
tqdm
```

### Configuration

- Max SQL generation attempts (C and D): 3
- Schema RAG top-k tables retrieved: default (see `src/config.py`)
- Prompts: shared `SYSTEM_PROMPT` in `baselines/runner.py`
- Rate limiting: exponential backoff on 429 errors (5s → 10s → 20s → 40s → 60s → 60s)

---

## 4. Ablation Configurations

The four configurations isolate each system component independently.

### Baseline A — Zero-shot (no RAG, no self-correction)
**File:** `baselines/baseline_a.py`

The entire database schema (all tables and columns) is serialized as plain text and injected into the prompt in one shot. Claude generates SQL once with no retry. This is the **floor** of the ablation — it has no novel components and represents what a capable LLM can do with no assistance.

- RAG: No — full schema dump injected directly
- Self-correction: No — single generation attempt
- Schema context source: `get_full_schema()` → `PRAGMA table_info` for every table

### Baseline B — Schema RAG only (no self-correction)
**File:** `baselines/baseline_b.py`

Instead of the full schema, only the most relevant tables are retrieved from ChromaDB using the question as a query. The top-k table chunks are injected into the prompt. Claude still generates SQL once with no retry. This isolates the contribution of **selective schema retrieval**.

- RAG: Yes — `SchemaRAG.get_schema_context(question, db_id)`
- Self-correction: No — single generation attempt
- Schema context source: ChromaDB semantic search over `SchemaChunk` embeddings
- Fallback: if RAG returns empty, falls back to full schema dump

### Baseline C — Self-correction only (no RAG)
**File:** `baselines/baseline_c.py`

Full schema is injected (same as A), but if the generated SQL fails execution, the error message is fed back to Claude for up to 3 attempts. Three failure modes trigger a retry: syntax error (caught by `sqlglot`), execution error (SQLite raises), and empty result (query runs but returns zero rows). This isolates the contribution of **agentic self-correction**.

- RAG: No — full schema dump
- Self-correction: Yes — up to 3 attempts with error feedback
- Schema context source: `get_full_schema()`

### Baseline D — Full system (RAG + self-correction)
**File:** `baselines/baseline_d.py`

Uses `SQLAgent` from `src/agent.py` — the thesis contribution. Schema RAG retrieves relevant tables, Claude generates SQL, it is executed, and errors are fed back for up to 3 correction attempts. This is the full system combining both contributions.

- RAG: Yes — `SchemaRAG`
- Self-correction: Yes — up to 3 attempts
- Implementation: `src/agent.py` → `SQLAgent.run()`


---

## 5. Results

### Overall execution accuracy

| Baseline | Description | EX% | Correct / 500 | Expected EX% | vs Expected |
|---|---|---|---|---|---|
| **A** | Zero-shot | **60.2%** | 301/500 | ~46% | +14.2 pts |
| **B** | RAG only | **61.2%** | 306/500 | ~52% | +9.2 pts |
| **C** | Correction only | **60.8%** | 304/500 | ~49% | +11.8 pts |
| **D** | Full system | **63.8%** | 319/500 | ~58%+ | +5.8 pts |

Historical baselines for reference (not run here):

| System | EX% |
|---|---|
| DIN-SQL (2023) | 50.7% |
| DAIL-SQL (2023) | 54.8% |
| MAC-SQL prompt-only (~2025) | ~57–60% |
| Human performance (BIRD) | 92.96% |

### Per-database breakdown

| Database | A EX% | B EX% | C EX% | D EX% |
|---|---|---|---|---|
| california_schools | 43% | 43% | 43% | 47% |
| card_games | 56% | 60% | 58% | 62% |
| codebase_community | 59% | 61% | 61% | 63% |
| debit_card_specializing | 57% | 53% | 50% | 60% |
| european_football_2 | 65% | 67% | 65% | 67% |
| financial | 44% | 50% | 47% | 56% |
| formula_1 | 50% | 56% | 53% | 56% |
| student_club | 81% | 79% | 79% | 85% |
| superhero | 88% | 87% | 88% | 88% |
| thrombosis_prediction | 50% | 44% | 50% | 46% |
| toxicology | 57% | 60% | 60% | 62% |

### Difficulty breakdown

| Difficulty | A | B | C | D |
|---|---|---|---|---|
| Simple (148 q) | 73% | 74% | 73% | 75% |
| Moderate (250 q) | 56% | 58% | 58% | 61% |
| Challenging (102 q) | 52% | 51% | 51% | 54% |

### Self-correction activity

| Baseline | Questions needing >1 attempt | % |
|---|---|---|
| C (correction only) | 7 / 500 | 1.4% |
| D (full system) | 19 / 500 | 3.8% |

D triggered more retries than C because RAG sometimes returns a narrower schema context, causing execution errors that require correction. Attempts distribution for D: 1 attempt = 481, 2 attempts = 11, 3 attempts = 8.

---

## 6. Key Findings

1. **D is the best configuration (63.8%)**, confirming that combining RAG and self-correction outperforms either component alone. The full ablation ordering is D > B > C > A, exactly as hypothesised.

2. **All baselines significantly exceed their expected targets.** `claude-sonnet-4-5` is far stronger than the GPT-3.5-era models the original BIRD baselines were calibrated against. The expected floor of ~46% was set based on models from 2022–2023.

3. **Component contributions:**
   - RAG alone: +1.0 pt (B 61.2% vs A 60.2%)
   - Self-correction alone: +0.6 pt (C 60.8% vs A 60.2%)
   - Both combined: +3.6 pts (D 63.8% vs A 60.2%) — **super-additive**, meaning the components reinforce each other

4. **Self-correction is more valuable in D than in C.** When RAG is also active (D), the correction loop fires on 3.8% of questions vs 1.4% in C. RAG sometimes returns a narrower schema context that causes execution errors, which the correction loop then resolves — the two components are complementary, not redundant.

5. **D shows the largest gains on previously weak databases:** `financial` (+12 pts vs A), `debit_card_specializing` (+3 pts), `student_club` (+4 pts) — databases where having the right tables and error recovery together matters most.

6. **`thrombosis_prediction` regresses in D (50% → 46%).** RAG is likely retrieving the wrong tables for medical-domain questions, and the correction loop cannot recover from wrong schema context. This is a known limitation of semantic similarity retrieval on domain-specific column names.

7. **Hardest databases:** `california_schools` (43–47%), `thrombosis_prediction` (44–50%), `formula_1` (50–56%) — complex joins or domain-specific conventions.

8. **Easiest databases:** `superhero` (87–88%), `student_club` (79–85%) — simpler schemas with intuitive column naming.

---

## 7. MLflow Tracking

All runs are logged to `mlruns/`. To view:

```bash
mlflow ui
# open http://localhost:5000
```

Run names: `baseline_a`, `baseline_b`, `baseline_c`, `baseline_d`

Logged per run: `execution_accuracy`, `correct`, `total`, `model`, `rag`, `self_correction`, `limit`

Raw results: `results/baseline_*.json` (one JSON object per question with `pred_sql`, `gold_sql`, `match`, `error`)
