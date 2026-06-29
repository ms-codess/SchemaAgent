# SchemaAgent — Ablation Study Results
_Full BIRD Mini-Dev run: 500 questions, 11 databases — 2026-04-12_
_See EVALUATION_RUN_2026-04-12.md for full per-question logs and error analysis._

| Config | RAG | Self-Correction | EX%    | Correct | Total |
|--------|-----|-----------------|--------|---------|-------|
| A      | No  | No              | 60.20% | 301     | 500   |
| B      | Yes | No              | 61.20% | 306     | 500   |
| C      | No  | Yes             | 60.80% | 304     | 500   |
| D      | Yes | Yes             | **63.80%** | **319** | 500 |

## Comparison Against Baselines

| System | EX%   | Notes |
|--------|-------|-------|
| Zero-shot Claude (config A, ours) | 60.20% | Our floor |
| DIN-SQL (2023) | 50.72% | Historical |
| DAIL-SQL (2023) | 54.76% | Historical |
| MAC-SQL prompt-only | ~57–60% | **Our primary target** |
| **SchemaAgent full (config D, ours)** | **63.80%** | **Beats target** |
| Agentar-Scale-SQL (fine-tuned SOTA) | 81.67% | Not our competition |

## Key Findings

- **RAG alone** (+1.0pp vs A): Schema-aware retrieval provides modest but consistent improvement.
- **Self-correction alone** (+0.6pp vs A): The retry loop recovers from execution errors at marginal cost (avg 1.21 attempts).
- **Combined system** (+3.6pp vs A, +2.6pp vs B): RAG + self-correction interact super-additively — the full system beats both ablations by more than either ablation beats the baseline.
- Config D beats MAC-SQL prompt-only (~57–60%) by ~4–7pp with no fine-tuning.

## Target
Beat MAC-SQL prompt-only (~57–60% EX). **Achieved at 63.80%.** Floor: zero-shot Claude (~60.2%).
