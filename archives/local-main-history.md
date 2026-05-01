# Local Main History Archive

This branch is a publishable manifest for the local-only branch
`archive/local-main-history` in the primary checkout at
`C:\Users\msmirani\Downloads\SchemaAgent`.

The raw branch could not be pushed to GitHub because its unpublished history
contains benchmark artifacts larger than GitHub's normal Git object limit.

## Local branch tip

- Branch: `archive/local-main-history`
- Tip commit: `e4b993b`
- Subject: `document thesis scope and add BirdDescRAG tests`

## Local-only commits on that branch

```text
e4b993b document thesis scope and add BirdDescRAG tests
da5220d eval
1e859b3 Implement feature X to enhance user experience and optimize performance
9ca0216 eval
958bf0e evaluation on BIRD mini dev
69d1756 add OpenRouter free Qwen option, fill in Claude eval results
f516930 add full documentation and dual Qwen provider support
5c41b32 switch Qwen to HuggingFace InferenceClient, remove openai dependency
da35a65 add Qwen2.5-Coder-32B support, fix BIRD paths, clean eval results
960c5e8 update UI visibility settings, add error and output log files, and remove unused rag module
```

## Push blocker

These unpublished objects exceed GitHub's standard size limit:

```text
1042548099  _bird_data/minidev/MINIDEV_mysql/BIRD_dev.sql
1001882533  _bird_data/minidev/MINIDEV_postgresql/BIRD_dev.sql
800943648   _bird_minidev.zip
597754880   _bird_data/minidev/MINIDEV/dev_databases/european_football_2/european_football_2.sqlite
481419264   _bird_data/minidev/MINIDEV/dev_databases/codebase_community/codebase_community.sqlite
261820416   _bird_data/minidev/MINIDEV/dev_databases/card_games/card_games.sqlite
```

## Notes

- The actual local branch remains preserved in the original checkout.
- Remote publishable replacements for active code changes were pushed separately on:
  - `origin/chore/claude-only-cleanup`
  - `origin/fix/pytest-clean`
