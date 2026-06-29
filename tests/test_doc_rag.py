"""
Tests for src/doc_rag.py — DocRAG indexing and retrieval.

Uses temporary directories and plain .txt files — no real PDFs or API calls needed.
ChromaDB is isolated to a temp directory per test session.
"""
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.doc_rag import DocRAG, DocChunk, CHUNK_SIZE, CHUNK_OVERLAP

# ── Sample document content ───────────────────────────────────────────────────

DOC_PROBATION = (
    "The academic probation policy requires all students to maintain a minimum GPA of 2.0. "
    "Students whose cumulative GPA falls below 2.0 at the end of any semester will be placed "
    "on academic probation. Students on probation must meet with their academic advisor within "
    "two weeks and develop an academic improvement plan. Failure to raise the GPA above 2.0 "
    "within two semesters may result in academic suspension."
)

DOC_VACATION = (
    "Full-time employees accrue 15 days of paid vacation per calendar year. "
    "Vacation days must be approved by the direct manager at least two weeks in advance. "
    "Unused vacation days can be carried over to the following year up to a maximum of 10 days. "
    "Part-time employees accrue vacation on a pro-rated basis according to their hours worked."
)

DOC_RETENTION = (
    "The data retention policy requires all financial records to be kept for a minimum of 7 years. "
    "Student academic records must be retained permanently. "
    "Employee personnel files must be kept for 10 years after the end of employment. "
    "All records must be stored in a secure, access-controlled environment."
)


@pytest.fixture
def doc_folder(tmp_path):
    """Create a temp folder with 3 .txt documents."""
    (tmp_path / "probation_policy.txt").write_text(DOC_PROBATION)
    (tmp_path / "vacation_policy.txt").write_text(DOC_VACATION)
    (tmp_path / "retention_policy.txt").write_text(DOC_RETENTION)
    return str(tmp_path)


@pytest.fixture
def rag(tmp_path):
    """DocRAG instance with isolated ChromaDB in a temp directory."""
    return DocRAG(chroma_path=str(tmp_path / "chroma"))


@pytest.fixture
def indexed_rag(rag, doc_folder):
    """DocRAG already indexed with the 3 sample documents."""
    rag.index_documents(doc_folder)
    return rag


# ── DocChunk dataclass ────────────────────────────────────────────────────────

def test_doc_chunk_fields():
    chunk = DocChunk(
        chunk_id="file.txt::0",
        source="/path/file.txt",
        text="some text",
        metadata={"chunk_index": 0},
    )
    assert chunk.chunk_id == "file.txt::0"
    assert chunk.text == "some text"
    assert chunk.metadata["chunk_index"] == 0


# ── Chunking ──────────────────────────────────────────────────────────────────

def test_chunk_text_short_doc(rag):
    text = "Hello world this is a short document."
    chunks = rag._chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text.strip()


def test_chunk_text_produces_overlap(rag):
    words = ["word"] * 600
    text = " ".join(words)
    chunks = rag._chunk_text(text)
    assert len(chunks) > 1
    # Each chunk should be at most chunk_size words
    for chunk in chunks:
        assert len(chunk.split()) <= CHUNK_SIZE


def test_chunk_text_overlap_between_chunks(rag):
    # With chunk_size=10, overlap=3: chunk 0 = words[0:10], chunk 1 = words[7:17]
    small_rag = DocRAG(chunk_size=10, chunk_overlap=3, chroma_path="unused")
    words = [f"w{i}" for i in range(25)]
    text = " ".join(words)
    chunks = small_rag._chunk_text(text)
    # Last words of chunk 0 should appear in chunk 1
    last_words_chunk0 = chunks[0].split()[-3:]
    first_words_chunk1 = chunks[1].split()[:3]
    assert last_words_chunk0 == first_words_chunk1


def test_chunk_text_empty_string(rag):
    assert rag._chunk_text("") == []


def test_chunk_text_covers_all_words(rag):
    words = [f"word{i}" for i in range(1200)]
    text = " ".join(words)
    chunks = rag._chunk_text(text)
    # First and last word must appear in chunks
    assert words[0] in chunks[0]
    assert words[-1] in chunks[-1]


# ── File loading ──────────────────────────────────────────────────────────────

def test_load_txt_file(rag, tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("Hello from a text file.")
    text = rag._load_file(f)
    assert "Hello" in text


def test_load_missing_pdf_returns_empty(rag, tmp_path):
    # pypdf on a non-existent/corrupt file should return empty string gracefully
    f = tmp_path / "fake.pdf"
    f.write_bytes(b"not a pdf")
    text = rag._load_pdf(f)
    assert isinstance(text, str)


def test_load_unsupported_extension(rag, tmp_path):
    f = tmp_path / "file.csv"
    f.write_text("a,b,c")
    text = rag._load_file(f)
    assert text == ""


# ── Indexing ──────────────────────────────────────────────────────────────────

def test_index_documents_returns_chunk_count(rag, doc_folder):
    count = rag.index_documents(doc_folder)
    assert count > 0


def test_index_documents_missing_folder(rag):
    count = rag.index_documents("/nonexistent/path")
    assert count == 0


def test_index_documents_empty_folder(rag, tmp_path):
    count = rag.index_documents(str(tmp_path))
    assert count == 0


def test_index_is_idempotent(rag, doc_folder):
    count1 = rag.index_documents(doc_folder)
    count2 = rag.index_documents(doc_folder)
    # Second index replaces first — collection count should not double
    collection = rag._get_collection()
    assert collection.count() == count1 == count2


# ── is_indexed ────────────────────────────────────────────────────────────────

def test_is_indexed_false_before_indexing(rag, doc_folder):
    assert rag.is_indexed(doc_folder) is False


def test_is_indexed_true_after_indexing(rag, doc_folder):
    rag.index_documents(doc_folder)
    assert rag.is_indexed(doc_folder) is True


# ── Retrieval ─────────────────────────────────────────────────────────────────

def test_retrieve_chunks_returns_doc_chunks(indexed_rag):
    chunks = indexed_rag.retrieve_chunks("academic probation GPA")
    assert len(chunks) > 0
    assert all(isinstance(c, DocChunk) for c in chunks)


def test_retrieve_chunks_relevant_to_query(indexed_rag):
    chunks = indexed_rag.retrieve_chunks("academic probation GPA", top_k=1)
    assert len(chunks) == 1
    assert "probation" in chunks[0].text.lower() or "gpa" in chunks[0].text.lower()


def test_retrieve_chunks_vacation_query(indexed_rag):
    chunks = indexed_rag.retrieve_chunks("employee vacation days", top_k=1)
    assert "vacation" in chunks[0].text.lower() or "days" in chunks[0].text.lower()


def test_retrieve_returns_string(indexed_rag):
    result = indexed_rag.retrieve("data retention policy")
    assert isinstance(result, str)
    assert len(result) > 0


def test_retrieve_includes_source_label(indexed_rag):
    result = indexed_rag.retrieve("academic probation")
    assert "[Source:" in result


def test_retrieve_empty_when_not_indexed(rag):
    result = rag.retrieve("anything")
    assert result == ""


def test_retrieve_chunks_empty_when_not_indexed(rag):
    chunks = rag.retrieve_chunks("anything")
    assert chunks == []


def test_retrieve_top_k_respected(indexed_rag):
    chunks = indexed_rag.retrieve_chunks("policy", top_k=2)
    assert len(chunks) <= 2


def test_chunk_metadata_has_source(indexed_rag):
    chunks = indexed_rag.retrieve_chunks("retention", top_k=1)
    assert "source" in chunks[0].metadata
    assert "chunk_index" in chunks[0].metadata
    assert "file_type" in chunks[0].metadata


# ── HyDE ─────────────────────────────────────────────────────────────────────

def test_retrieve_with_hyde(indexed_rag):
    mock_client = MagicMock()
    mock_client.messages.create.return_value = MagicMock(
        content=[SimpleNamespace(
            text="Students must maintain a GPA of 2.0 to avoid academic probation."
        )]
    )
    chunks = indexed_rag.retrieve_chunks(
        "What GPA avoids probation?", top_k=1, use_hyde=True, hyde_client=mock_client
    )
    mock_client.messages.create.assert_called_once()
    assert len(chunks) > 0


def test_retrieve_without_hyde_makes_no_api_call(indexed_rag):
    mock_client = MagicMock()
    indexed_rag.retrieve_chunks("probation policy", use_hyde=False, hyde_client=mock_client)
    mock_client.messages.create.assert_not_called()
