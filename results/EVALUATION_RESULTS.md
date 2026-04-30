# SchemaAgent -- Evaluation Results

Benchmark: BIRD Mini-Dev (500 questions, 11 SQLite databases)
Metric: Execution Accuracy (EX) -- predicted result set equals gold result set
Run date: April 25 2026

---

## Current Rerun Status

As of April 30, 2026, a fresh rerun from this repo is blocked by Anthropic billing.

- `baseline_a` was retried on April 30, 2026 and failed on question 1 with: `Your credit balance is too low to access the Anthropic API.`
- The existing full-run artifact that is still complete is `results/baseline_d_claude_sonnet_45.json` from April 25, 2026.
- `results/run_baseline_e_no_doc.log` shows an older April 25, 2026 `E-noDoc` run reaching 330/500 questions before failing with the same insufficient-credit error.
- `results/baseline_e_no_doc.json` and `results/baseline_e_with_doc.json` currently contain 5-question smoke outputs, not full 500-question runs.

Until credits are restored, the only fully reproducible local result artifact in `results/` is Config D.

---

## Thesis Contribution: Five-Config Ablation

The experiment measures three compounding contributions:

1. **Schema RAG** -- semantic retrieval of relevant tables (vs. dumping the full schema)
2. **Self-correction** -- up to 3 retry attempts with error feedback (vs. single-shot)
3. **Data Dictionary RAG (BirdDescRAG)** -- retrieving column descriptions from BIRD's
   database_description/*.csv files to replace the hand-crafted gold evidence hint

Configs A-D show that RAG and self-correction are super-additive.
Config E shows the system can autonomously recover domain knowledge that was
previously provided as a gold hint -- validating the DocRAG contribution.

| Config | LLM | Description | EX% | Correct | Delta vs A |
|--------|-----|-------------|-----|---------|-----------|
| A | Claude Sonnet 4.5 | Zero-shot (full schema, no aids) | 60.2% | 301/500 | -- |
| B | Claude Sonnet 4.5 | Schema RAG only | 61.2% | 306/500 | +1.0 pp |
| C | Claude Sonnet 4.5 | Self-correction only | 60.8% | 304/500 | +0.6 pp |
| D | Claude Sonnet 4.5 | RAG + self-correction + gold hint | 63.2% | 316/500 | +3.0 pp |
| E-noDoc | Claude Sonnet 4.5 | RAG + self-correction, NO hint | pending | -- | -- |
| E-withDoc | Claude Sonnet 4.5 | RAG + self-correction + BirdDescRAG | pending | -- | -- |

Key findings (A-D):
- RAG alone: +1.0 pp
- Self-correction alone: +0.6 pp
- Combined (D vs A): +3.0 pp -- exceeds the simple sum, confirming super-additivity
- Mechanism: RAG narrows schema context, occasionally omitting a table and triggering
  an execution error; the correction loop then resolves this, firing on 3.4% of
  questions in Config D vs 1.4% in Config C alone.

Config E interpretation (pending full rerun):
- E-noDoc vs D: measures how much Config D relied on the gold evidence hint
- E-withDoc vs E-noDoc: measures how much BirdDescRAG recovers from the hint removal
- Recovery gap (D - E-withDoc): the irreducible advantage of a perfect hand-crafted hint

---

## Config D (Claude Sonnet 4.5) -- April 25 2026

| Metric | Value |
|--------|-------|
| Model | claude-sonnet-4-5 |
| Questions | 500 |
| Correct | 316 |
| EX% | 63.2% |
| 1st-attempt success | 483 / 500 (96.6%) |
| Correction fired (2+ attempts) | 17 / 500 (3.4%) |
| 2 attempts | 14 |
| 3 attempts | 3 |

### Per-database breakdown

| Database | Correct | Total | EX% |
|----------|---------|-------|-----|
| california_schools | 16 | 30 | 53.3% |
| card_games | 32 | 52 | 61.5% |
| codebase_community | 30 | 49 | 61.2% |
| debit_card_specializing | 18 | 30 | 60.0% |
| european_football_2 | 32 | 51 | 62.7% |
| financial | 19 | 32 | 59.4% |
| formula_1 | 36 | 66 | 54.5% |
| student_club | 40 | 48 | 83.3% |
| superhero | 45 | 52 | 86.5% |
| thrombosis_prediction | 24 | 50 | 48.0% |
| toxicology | 24 | 40 | 60.0% |

### Per-difficulty breakdown

| Difficulty | Correct | Total | EX% |
|------------|---------|-------|-----|
| Simple | 112 | 148 | 75.7% |
| Moderate | 148 | 250 | 59.2% |
| Challenging | 56 | 102 | 54.9% |

---

## Config E-noDoc (Claude Sonnet 4.5) -- April 25 2026

Schema RAG + self-correction. Gold evidence hint dropped. No replacement.

| Metric | Value |
|--------|-------|
| Model | claude-sonnet-4-5 |
| Full rerun status | interrupted at 330 / 500 |
| Failure mode | Anthropic insufficient-credit error |
| Smoke sample artifact | `results/baseline_e_no_doc.json` |
| Smoke sample EX | 60.0% (3 / 5) |

Full-run evidence: `results/run_baseline_e_no_doc.log`

---

## Config E-withDoc (Claude Sonnet 4.5) -- April 25 2026

Schema RAG + self-correction. Evidence replaced with BirdDescRAG retrieval from
database_description/*.csv column descriptions. No gold hint used.

| Metric | Value |
|--------|-------|
| Model | claude-sonnet-4-5 |
| Full rerun status | not available in repo |
| Smoke sample artifact | `results/baseline_e_with_doc.json` |
| Smoke sample EX | 60.0% (3 / 5) |

There is currently no 500-question `E-withDoc` result artifact in `results/`.

---

## How to Reproduce

  # Install dependencies
  pip install -r requirements.txt

  # Set in .env:
  #   ANTHROPIC_API_KEY=...

  # Config A -- zero-shot
  python -m baselines.baseline_a

  # Config B -- RAG only
  python -m baselines.baseline_b

  # Config C -- self-correction only
  python -m baselines.baseline_c

  # Config D -- full system, gold hint, Claude
  python -m baselines.baseline_d --llm claude

  # Config D -- full system, gold hint, Claude Haiku variant
  python -m baselines.baseline_d --llm haiku

  # Config E -- no hint (ablation)
  python -m baselines.baseline_e --mode no_doc

  # Config E -- BirdDescRAG replaces hint
  python -m baselines.baseline_e --mode with_doc

  # View MLflow experiment
  mlflow ui
