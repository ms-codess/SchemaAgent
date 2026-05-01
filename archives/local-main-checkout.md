# Local Main Checkout Archive

This branch is a publishable manifest for the local-only branch
`wip/local-main-checkout` in the primary checkout at
`C:\Users\msmirani\Downloads\SchemaAgent`.

The raw branch could not be pushed to GitHub because its underlying unpublished
history inherits the same oversized benchmark artifacts as
`archive/local-main-history`.

## Local branch tip

- Branch: `wip/local-main-checkout`
- Tip commit: `9e4b0be`
- Subject: `wip: preserve local main checkout state`

## What this checkout snapshot adds on top of `archive/local-main-history`

```text
CLAUDE.md
README.md
app.py
baselines/baseline_d.py
baselines/baseline_e.py
docs/ARCHITECTURE.md
requirements.txt
results/EVALUATION_RESULTS.md
results/run_baseline_a_20260430.log
results/run_baseline_e_no_doc.log
src/config.py
src/fusion.py
src/llm_client.py
_bird_mini_dev (gitlink/submodule pointer change)
```

## Diffstat for the preserved checkout commit

```text
13 files changed, 1005 insertions(+), 346 deletions(-)
```

## Notes

- This local checkout snapshot mostly contains the Claude-only cleanup and
  evaluation-status documentation that were later published cleanly on:
  - `origin/chore/claude-only-cleanup`
  - `origin/fix/pytest-clean`
- The local branch still exists in the original checkout for exact recovery.
- The raw branch could not be pushed because GitHub rejected the oversized
  benchmark blobs reachable from its base history.
