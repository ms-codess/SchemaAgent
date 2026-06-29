# SchemaAgent — Claude Project Context

## What This Project Is

An agentic RAG system that converts natural language questions into SQL and retrieves answers from company documents. Evaluated on the BIRD Text-to-SQL benchmark (bird-bench.github.io).

**One-line pitch:** Talk to any database in plain English — with document awareness and self-correction.

---

## Research Positioning (Critical — Read Before Every Session)

### The Gap We Fill

Existing agentic Text-to-SQL systems (MAC-SQL 2025, CHASE-SQL 2024) address structured database querying but ignore unstructured organizational knowledge. BIRD-Interact (ICLR 2026 Oral) introduces a knowledge base component but does not evaluate hybrid retrieval across structured AND unstructured sources in a unified conversational interface. This thesis fills that gap.

### What We Are NOT Claiming

- We do NOT claim to beat fine-tuned SOTA systems (Agentar-Scale-SQL: 81.67% EX)
- We do NOT claim to beat MAC-SQL on pure SQL accuracy
- We DO claim to be the first to formally evaluate hybrid structured + unstructured retrieval on BIRD

### Baselines We Compare Against

| System | EX% | Role in thesis |
|---|---|---|
| Zero-shot Claude (no RAG) | ~46% | Our floor — must beat |
| DIN-SQL (2023) | 50.7% | Historical reference |
| DAIL-SQL (2023) | 54.8% | Historical reference |
| MAC-SQL prompt-only (no fine-tuning) | ~57–60% | Our real target |

---

## System Architecture — Five Layers

```
Layer 0: Streamlit UI          — chat interface, DB upload, doc upload
Layer 1: AI Agent              — Claude API, chain-of-thought, tool orchestration
Layer 2: Intent Router         — classifies: DB query / doc search / hybrid
Layer 3A: Schema RAG           — ChromaDB of table/field definitions (BIRD + production DBs)
Layer 3B: Document RAG         — ChromaDB of PDFs, Word docs, policy files
Layer 4: Text-to-SQL Engine    — SQL generation + self-correction loop (max 3 attempts)
Layer 5: Database              — SQLite (BIRD research) / Oracle / PostgreSQL (production)
```

---

## Three Novel Contributions

1. **Schema-aware RAG** — selective table retrieval vs full schema injection. Measured independently in ablation.
2. **Agentic self-correction loop** — error-informed multi-turn retry. Measured independently in ablation.
3. **Hybrid structured + unstructured retrieval** — first formal evaluation of DB + document fusion on BIRD. No prior paper does this.

---

## Ablation Study Design

| System | RAG | Self-correction | Expected EX% |
|---|---|---|---|
| A — Baseline | No | No | ~46% |
| B — RAG only | Yes | No | ~52% |
| C — Correction only | No | Yes | ~49% |
| D — Full system (ours) | Yes | Yes | ~58%+ |

All four run on the same 500 BIRD Mini-Dev questions. Log everything to MLflow.

---

## Technology Stack

```
LLM backbone:        Claude API (claude-sonnet-4-5)
Vector store:        ChromaDB (local)
Embeddings:          all-MiniLM-L6-v2 (free, runs locally)
SQL validation:      sqlglot
SQL execution:       SQLite (BIRD) / SQLAlchemy (production)
Experiment tracker:  MLflow
UI:                  Streamlit
Deployment:          Railway (free tier)
Language:            Python 3.11
DB (production):     Oracle via python-oracledb + SQLAlchemy
```

---

## Project File Structure

```
schemaagent/
├── CLAUDE.md                    ← this file
├── PROJECT_PLAN.md              ← full 4-week plan
├── requirements.txt
├── app.py                       ← Streamlit UI (Phase 8)
├── config/
│   ├── bird_sqlite.yaml         ← BIRD research config
│   ├── uottawa_oracle.yaml      ← uOttawa Oracle production config
│   └── example.yaml             ← template for new clients
├── src/
│   ├── schema/
│   │   ├── __init__.py          ← exports SchemaRAG facade
│   │   ├── serializer.py        ← Phase 3: SQLite → SchemaChunk list
│   │   ├── indexer.py           ← Phase 3: chunks → ChromaDB
│   │   └── retriever.py         ← Phase 3: question + db_id → top-k tables
│   ├── doc_rag.py               ← Phase 6: document indexing + retrieval
│   ├── agent.py                 ← Phase 5: self-correcting agent loop
│   ├── router.py                ← Phase 7: intent classifier
│   ├── evaluator.py             ← Phase 9: BIRD EX% evaluation
│   ├── config.py                ← model names, paths, shared constants
│   └── utils.py                 ← BIRD loader, DB connector, helpers
├── baselines/
│   ├── baseline_a.py            ← no RAG, no correction
│   ├── baseline_b.py            ← RAG only
│   └── baseline_c.py            ← correction only
├── data/
│   ├── .gitkeep                 ← BIRD files go here, NOT committed
│   └── bird/
│       ├── mini_dev_sqlite.json ← 500 questions + gold SQL (NOT committed)
│       └── dev_databases/       ← 11 SQLite databases (NOT committed)
├── tests/
│   ├── test_schema_rag.py
│   ├── test_doc_rag.py
│   └── test_router.py
├── thesis/
│   ├── chapter1_draft.md
│   ├── chapter2_draft.md
│   ├── chapter3_draft.md
│   ├── chapter4_draft.md
│   └── chapter5_draft.md
└── notes/
    └── papers.md                ← paper notes using the template
```

---

## Git Branch Strategy

```
main                      ← stable only, never broken
feature/schema-rag        ← Phase 3
feature/baselines         ← Phase 4
feature/self-correction   ← Phase 5
feature/doc-rag           ← Phase 6
feature/intent-router     ← Phase 7
feature/hybrid-fusion     ← Phase 8
feature/ui                ← Phase 8
feature/evaluation        ← Phase 9
```

Merge rule: only merge to main when the phase is fully working and tested.

---

## Paper Notes Template (use for every paper)

```markdown
## [Title] — [Authors, Year]
Link: arxiv.org/abs/...

Problem they solve:

Their approach:

Key results / numbers:

How I use this in my thesis:
- I cite this in section X because...
- My system improves on / extends this by...

Terms I didn't know:
```

---

## Reading List — In Order

### Week 1 (read before coding)
1. RAG — Lewis et al. 2020 — `arxiv.org/abs/2005.11401` ✅ DONE
2. BIRD — Li et al. 2023 — `arxiv.org/abs/2305.03111` ✅ DONE
3. MAC-SQL — Wang et al. 2025 (ACL) — your closest related work
4. CHASE-SQL — Pourreza et al. 2024 — `arxiv.org/abs/2411.00841`

### Week 2 (read while building agent)
5. BIRD-Interact — BIRD Team 2025 (ICLR 2026 Oral) — most relevant recent work
6. ReAct — Yao et al. 2022 — `arxiv.org/abs/2210.03629`
7. Self-RAG — Asai et al. 2023 — `arxiv.org/abs/2310.11511`

### Week 3 (read while building doc RAG)
8. HyDE — Gao et al. 2022 — `arxiv.org/abs/2212.10496`

### Historical reference (skim only, ~20 min each)
- DIN-SQL — Pourreza 2023 — `arxiv.org/abs/2304.11015`
- DAIL-SQL — Gao 2023 — `arxiv.org/abs/2308.15363`

---

## Production Extension — Oracle DB

The system connects to Oracle (uOttawa PeopleSoft) via SQLAlchemy.
Connection string: `oracle+cx_oracle://user:pass@host:port/service`
Safety requirements:
- Reject any non-SELECT statement before execution (sqlglot validation)
- Append FETCH FIRST 1000 ROWS ONLY to all queries
- Use read-only service account credentials only

---

## Current Status

### Code — COMPLETE
- [x] RAG paper read
- [x] BIRD paper read
- [x] Project structure created in VSCode
- [x] All branches created
- [x] BIRD Mini-Dev downloaded (500 questions, 11 databases → data/bird/)
- [x] src/schema/ built (serializer, indexer, retriever) — tests passing
- [x] src/utils.py — load_bird_questions, get_db_connection, execute_sql, extract_sql_from_response, SYSTEM_PROMPT, build_user_message, get_full_schema
- [x] Baseline A (zero-shot) — built and tested
- [x] Baseline B (RAG only) — built and tested
- [x] Baseline C (correction only) — built and tested
- [x] Baseline D (full system) — built and tested
- [x] SQL Agent with self-correction loop (src/agent.py) — built and tested
- [x] Document RAG (src/doc_rag.py) — PDF/DOCX/TXT, HyDE — built and tested
- [x] Intent Router (src/router/) — DB / doc / hybrid classification — built and tested
- [x] Hybrid Fusion (src/fusion.py) — SQL + doc synthesis — built and tested
- [x] Evaluator (src/evaluator.py) — 4 ablation configs on BIRD — built and tested
- [x] Streamlit UI (app.py) — chat interface, DB upload, doc upload — built
- [x] Full BIRD evaluation run — 500 questions, 11 databases
- [x] 127/127 tests passing
- [x] railway.toml added for deployment

### Results (BIRD Mini-Dev, 500 questions)
| Config | EX% | vs target |
|--------|-----|-----------|
| A — Baseline | 60.2% | floor |
| B — RAG only | 61.2% | +1.0pp |
| C — Correction only | 60.8% | +0.6pp |
| D — Full system | **63.8%** | **beats MAC-SQL prompt-only** |

### Reading — IN PROGRESS
- [x] RAG — Lewis et al. 2020
- [x] BIRD — Li et al. 2023
- [ ] MAC-SQL — Wang et al. 2025 (ACL) ← read next
- [ ] CHASE-SQL — Pourreza et al. 2024
- [ ] BIRD-Interact — BIRD Team 2025 (ICLR 2026 Oral)
- [ ] ReAct — Yao et al. 2022
- [ ] Self-RAG — Asai et al. 2023
- [ ] HyDE — Gao et al. 2022

### Thesis — NOT STARTED
- [ ] Chapter 1: Introduction
- [ ] Chapter 2: Literature Review
- [ ] Chapter 3: System Design
- [ ] Chapter 4: Experiments
- [ ] Chapter 5: Conclusion
- [ ] Abstract

### Remaining tasks
- [ ] Read remaining 6 papers (MAC-SQL first — closest related work)
- [ ] Write thesis chapters (one paragraph per day minimum)
- [ ] Deploy to Railway (railway.toml is ready — push to GitHub then connect)

### Branch build order (all complete)
| # | Branch | Status |
|---|---|---|
| 0 | `feature/schema-rag` | ✅ Done |
| — | `feature/baselines` | ✅ Done (A/B/C/D) |
| 1 | `feature/self-correction` | ✅ Done |
| 2 | `feature/doc-rag` | ✅ Done |
| 3 | `feature/intent-router` | ✅ Done |
| 4 | `feature/hybrid-fusion` | ✅ Done |
| 5 | `feature/evaluation` | ✅ Done |
| 6 | `feature/ui` | ✅ Done |

---

## Thesis Structure

```
Abstract (250 words — write last)
Chapter 1: Introduction (4–6 pages) — problem, gap, 3 contributions
Chapter 2: Literature Review (8–10 pages) — 8 papers + gap statement
Chapter 3: System Design (10–12 pages) — 5 layers, architecture diagram, prompts
Chapter 4: Experiments (8–10 pages) — ablation table, BIRD results, error analysis
Chapter 5: Conclusion (3–4 pages) — findings, limitations, future work
```

Target length: 40–55 pages excluding references.

---

## Daily Work Pattern

- Morning: theory (read paper using 3-pass method)
- Midday: thesis writing (one paragraph minimum, every day)
- Afternoon: code (implement the current phase)

Never code before reading. Never skip thesis writing. One paragraph per day = thesis done by day 28.

---

## Key Numbers to Know at All Times

- BIRD Mini-Dev size: 500 questions (11 SQLite databases) — what we actually run
- BIRD full dev set: 1,534 questions, 95 databases — referenced in thesis for context
- Our target EX%: beat MAC-SQL prompt-only (~57–60%)
- Zero-shot floor: ~46% EX
- Human performance on BIRD: 92.96%
- SOTA (Agentar-Scale-SQL, fine-tuned): 81.67% EX — NOT our competition

---

## Goal

Beat MAC-SQL prompt-only on BIRD. Ship a live demo. Land a remote ML Engineer role by mid-2026.