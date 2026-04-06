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

# BIRD data paths (can be overridden via env vars)
BIRD_DB_ROOT = os.environ.get(
    "BIRD_DB_ROOT",
    str(PROJECT_ROOT / "data" / "bird" / "dev_databases"),
)
BIRD_DEV_JSON = os.environ.get(
    "BIRD_DEV_JSON",
    str(PROJECT_ROOT / "data" / "bird" / "mini_dev_sqlite.json"),
)
