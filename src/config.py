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
    str(PROJECT_ROOT / "_bird_data" / "minidev" / "MINIDEV" / "dev_databases"),
)
BIRD_DEV_JSON = os.environ.get(
    "BIRD_DEV_JSON",
    str(PROJECT_ROOT / "_bird_data" / "minidev" / "MINIDEV" / "mini_dev_sqlite.json"),
)

# ── LLM provider registry ─────────────────────────────────────────────────────
# provider options:
#   "anthropic"    — anthropic SDK (ANTHROPIC_API_KEY)
#   "huggingface"  — huggingface_hub InferenceClient (HF_TOKEN, requires PRO for 32B)
#   "openai_compat"— openai-compatible REST endpoint (pip install openai)
#
# Qwen note: HuggingFace free tier depletes quickly on 32B models.
# Set TOGETHER_API_KEY and use the Together.ai entry for reliable evals.
LLM_OPTIONS: dict = {
    "Claude Sonnet 4.5": {
        "provider": "anthropic",
        "model":    "claude-sonnet-4-5",
    },
    "Claude Haiku 4.5": {
        "provider": "anthropic",
        "model":    "claude-haiku-4-5-20251001",
    },
    # OpenRouter — FREE tier (rate-limited), same 32B model, requires: pip install openai
    # Sign up free at openrouter.ai → copy API key → set OPENROUTER_API_KEY in .env
    "Qwen2.5-Coder-32B": {
        "provider":    "openai_compat",
        "model":       "qwen/qwen-2.5-coder-32b-instruct:free",
        "base_url":    "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
    },
    # HuggingFace Serverless — requires HF_TOKEN + HF PRO subscription for 32B
    "Qwen2.5-Coder-32B (HuggingFace PRO)": {
        "provider":    "huggingface",
        "model":       "Qwen/Qwen2.5-Coder-32B-Instruct",
        "api_key_env": "HF_TOKEN",
    },
    # Together.ai — pay-per-token, requires: pip install openai + TOGETHER_API_KEY
    "Qwen2.5-Coder-32B (Together.ai)": {
        "provider":    "openai_compat",
        "model":       "Qwen/Qwen2.5-Coder-32B-Instruct",
        "base_url":    "https://api.together.xyz/v1",
        "api_key_env": "TOGETHER_API_KEY",
    },
}

DEFAULT_LLM = "Claude Sonnet 4.5"
