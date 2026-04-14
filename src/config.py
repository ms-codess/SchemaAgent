import os
from pathlib import Path

# Project root — two levels up from this file (src/config.py → project root)
PROJECT_ROOT = Path(__file__).parent.parent

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ChromaDB
CHROMA_PATH = str(PROJECT_ROOT / "chroma_db")
SCHEMA_COLLECTION = "schema_tables"

# Schema RAG defaults
SAMPLE_ROWS = 3       # distinct non-null values to sample per column
MAX_COLUMNS = 20      # cap for very wide tables
TOP_K = 5             # tables returned per query

# ── Model routing ─────────────────────────────────────────────────────────────
# Centralise model names so switching costs one line, not a grep.
MODEL_SQL           = "claude-sonnet-4-5"           # SQL generation + self-correction
MODEL_ROUTER        = "claude-haiku-4-5-20251001"   # intent classification (cheap + fast)
MODEL_SYNTH_DB      = "claude-haiku-4-5-20251001"   # DB-only natural-language synthesis
MODEL_SYNTH_DOC     = "claude-haiku-4-5-20251001"   # doc-only natural-language synthesis
MODEL_SYNTH_HYBRID  = "claude-sonnet-4-5"           # hybrid fusion (needs reasoning)

# ── Token caps ─────────────────────────────────────────────────────────────────
MAX_TOKENS_SQL      = 500     # SQL generation — short, structured output
MAX_TOKENS_SYNTH    = 512     # DB / doc synthesis — one to a few sentences
MAX_TOKENS_FUSION   = 1024    # hybrid fusion — may need more reasoning room

# BIRD data paths (can be overridden via env vars)
BIRD_DB_ROOT = os.environ.get(
    "BIRD_DB_ROOT",
    str(PROJECT_ROOT / "data" / "bird" / "dev_databases"),
)
BIRD_DEV_JSON = os.environ.get(
    "BIRD_DEV_JSON",
    str(PROJECT_ROOT / "data" / "bird" / "mini_dev_sqlite.json"),
)
