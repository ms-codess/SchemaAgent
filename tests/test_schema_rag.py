"""
Tests for src/schema/ — serializer, indexer, retriever.

These tests use the BIRD Mini-Dev california_schools database when the local
benchmark payload is available. In the published repo that payload is optional,
so the module skips cleanly if it is missing.
"""
from pathlib import Path

import pytest

from src.schema.serializer import SchemaSerializer, SchemaChunk
from src.schema.indexer import SchemaIndexer
from src.schema.retriever import SchemaRetriever
from src.config import BIRD_DB_ROOT

DB_ROOT = Path(BIRD_DB_ROOT)
DB_PATH = str(DB_ROOT / "california_schools" / "california_schools.sqlite")
DB_ID = "california_schools"

if not Path(DB_PATH).exists():
    pytest.skip(
        "BIRD Mini-Dev SQLite payload not present locally; schema RAG tests require "
        "the optional benchmark download.",
        allow_module_level=True,
    )


# ── Serializer ────────────────────────────────────────────────────────────────

def test_serializer_returns_chunks():
    serializer = SchemaSerializer()
    chunks = serializer.serialize_database(DB_PATH, DB_ID)
    assert len(chunks) > 0, "Expected at least one table"
    for c in chunks:
        assert isinstance(c, SchemaChunk)
        assert c.db_id == DB_ID
        assert c.table_name
        assert c.text
        assert c.chunk_id == f"{DB_ID}::{c.table_name}"


def test_serializer_text_contains_table_name():
    serializer = SchemaSerializer()
    chunks = serializer.serialize_database(DB_PATH, DB_ID)
    for c in chunks:
        assert c.table_name in c.text


def test_serializer_missing_db_returns_empty():
    serializer = SchemaSerializer()
    result = serializer.serialize_database("/nonexistent/path.sqlite", "fake_db")
    assert result == []


def test_serializer_sample_values_in_text():
    serializer = SchemaSerializer(sample_rows=2)
    chunks = serializer.serialize_database(DB_PATH, DB_ID)
    # At least one chunk should contain "examples:" from non-empty columns
    texts_with_examples = [c for c in chunks if "examples:" in c.text]
    assert len(texts_with_examples) > 0


# ── Indexer ───────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_chroma(tmp_path):
    """Return a temporary ChromaDB path so tests don't pollute the real chroma_db/."""
    return str(tmp_path / "test_chroma")


def test_indexer_indexes_database(tmp_chroma):
    serializer = SchemaSerializer()
    chunks = serializer.serialize_database(DB_PATH, DB_ID)

    indexer = SchemaIndexer(chroma_path=tmp_chroma)
    count = indexer.index_database(chunks)
    assert count == len(chunks)


def test_indexer_is_idempotent(tmp_chroma):
    serializer = SchemaSerializer()
    chunks = serializer.serialize_database(DB_PATH, DB_ID)

    indexer = SchemaIndexer(chroma_path=tmp_chroma)
    count_first = indexer.index_database(chunks)
    count_second = indexer.index_database(chunks)  # upsert — should not grow

    # Verify collection size didn't double
    collection = indexer._get_collection()
    total = collection.count()
    assert total == count_first == count_second


def test_indexer_empty_chunks_returns_zero(tmp_chroma):
    indexer = SchemaIndexer(chroma_path=tmp_chroma)
    count = indexer.index_database([])
    assert count == 0


# ── Retriever ─────────────────────────────────────────────────────────────────

@pytest.fixture
def indexed_retriever(tmp_chroma):
    """Index california_schools, return a retriever pointed at the same chroma_path."""
    serializer = SchemaSerializer()
    chunks = serializer.serialize_database(DB_PATH, DB_ID)

    indexer = SchemaIndexer(chroma_path=tmp_chroma)
    indexer.index_database(chunks)

    return SchemaRetriever(chroma_path=tmp_chroma)


def test_retriever_returns_results(indexed_retriever):
    question = "What is the average SAT score for schools in Los Angeles?"
    chunks = indexed_retriever.retrieve(question, DB_ID, top_k=3)
    assert len(chunks) > 0


def test_retriever_scoped_to_db_id(tmp_chroma):
    """Results from one db_id must not include tables from another db_id."""
    card_db = str(DB_ROOT / "card_games" / "card_games.sqlite")

    serializer = SchemaSerializer()
    indexer = SchemaIndexer(chroma_path=tmp_chroma)

    # Index two databases
    indexer.index_database(serializer.serialize_database(DB_PATH, DB_ID))
    indexer.index_database(serializer.serialize_database(card_db, "card_games"))

    retriever = SchemaRetriever(chroma_path=tmp_chroma)
    chunks = retriever.retrieve("schools with high SAT scores", DB_ID, top_k=5)

    for c in chunks:
        assert c.db_id == DB_ID, f"Expected {DB_ID}, got {c.db_id}"


def test_retriever_relevant_table_in_top_k(indexed_retriever):
    """The 'schools' table should surface when asking about school names."""
    question = "List the names of all schools in the database"
    chunks = indexed_retriever.retrieve(question, DB_ID, top_k=5)
    table_names = [c.table_name.lower() for c in chunks]
    assert any("school" in t for t in table_names), f"Got: {table_names}"


def test_retriever_as_text_returns_string(indexed_retriever):
    text = indexed_retriever.retrieve_as_text(
        "How many students passed the SAT?", DB_ID, top_k=3
    )
    assert isinstance(text, str)
    assert len(text) > 0


def test_retriever_unindexed_db_returns_empty(tmp_chroma):
    retriever = SchemaRetriever(chroma_path=tmp_chroma)
    chunks = retriever.retrieve("any question", "nonexistent_db", top_k=3)
    assert chunks == []
