# SchemaAgent — Ablation Study Results
_Generated: 2026-04-07 18:23_

| Config | RAG | Self-Correction | EX% | Correct | Total |
|--------|-----|-----------------|-----|---------|-------|
| A | No | No | 80.00% | 4 | 5 |
| B | Yes | No | 100.00% | 5 | 5 |
| C | No | Yes | 100.00% | 5 | 5 |
| D | Yes | Yes | 100.00% | 5 | 5 |

## Target
Beat MAC-SQL prompt-only (~57–60% EX). Floor: zero-shot Claude (~46%).