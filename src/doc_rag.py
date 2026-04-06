"""
Document RAG — feature/doc-rag.

Indexes PDF and Word documents into ChromaDB and retrieves relevant
passages for a given question.

Pipeline:
    folder → load files → chunk text (500 words / 50 overlap)
           → embed with all-MiniLM-L6-v2 (local, free)
           → upsert into ChromaDB collection "doc_chunks"
    query  → embed question → cosine search → return top-k passages

No API calls are made during indexing or retrieval (unless use_hyde=True,
which generates a hypothetical answer via Claude before embedding).

Usage:
    rag = DocRAG()
    rag.index_documents("docs/")
    answer_context = rag.retrieve("What is the academic probation policy?")
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

DOC_COLLECTION = "doc_chunks"
CHUNK_SIZE = 500      # words per chunk
CHUNK_OVERLAP = 50    # word overlap between adjacent chunks
TOP_K_DOCS = 3

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

# ── DocChunk dataclass ────────────────────────────────────────────────────────

@dataclass
class DocChunk:
    chunk_id: str          # "{filename}::{chunk_index}"
    source: str            # original file path
    text: str              # raw passage text
    metadata: dict = field(default_factory=dict)


# ── DocRAG ────────────────────────────────────────────────────────────────────

class DocRAG:
    """
    Index and retrieve document passages using ChromaDB + local embeddings.

    Parameters
    ----------
    chroma_path : str
        Path to the ChromaDB persistent store (separate from schema RAG).
    collection_name : str
        ChromaDB collection name. Default "doc_chunks".
    model_name : str
        Sentence-transformer model for embeddings.
    chunk_size : int
        Approximate number of words per chunk.
    chunk_overlap : int
        Number of words shared between adjacent chunks.
    """

    def __init__(
        self,
        chroma_path: str = "chroma_db",
        collection_name: str = DOC_COLLECTION,
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._collection = None

    # ── Public interface ──────────────────────────────────────────────────────

    def index_documents(self, folder_path: str) -> int:
        """
        Index all supported documents in folder_path.
        Returns the total number of chunks indexed.
        Idempotent — re-indexing the same file replaces existing chunks.
        """
        folder = Path(folder_path)
        if not folder.exists():
            logger.warning("Document folder not found: %s", folder_path)
            return 0

        files = [
            f for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        if not files:
            logger.warning("No supported documents found in %s", folder_path)
            return 0

        total = 0
        for file_path in files:
            count = self._index_file(file_path)
            total += count
            logger.info("Indexed %s → %d chunks", file_path.name, count)

        logger.info("Total chunks indexed from %s: %d", folder_path, total)
        return total

    def retrieve_chunks(
        self,
        question: str,
        top_k: int = TOP_K_DOCS,
        use_hyde: bool = False,
        hyde_client=None,
    ) -> List[DocChunk]:
        """
        Retrieve the top-k most relevant DocChunks for a question.

        Parameters
        ----------
        use_hyde : bool
            If True, generate a hypothetical answer via Claude and embed that
            instead of the bare question (improves retrieval quality).
            Requires hyde_client (an anthropic.Anthropic instance).
        """
        collection = self._get_collection()
        if collection.count() == 0:
            logger.warning("No documents indexed. Call index_documents() first.")
            return []

        query_text = question
        if use_hyde and hyde_client is not None:
            query_text = self._generate_hypothetical_answer(hyde_client, question)
            logger.debug("HyDE query: %s", query_text[:80])

        results = collection.query(
            query_texts=[query_text],
            n_results=min(top_k, collection.count()),
        )

        chunks = []
        for chunk_id, doc, meta in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
        ):
            chunks.append(DocChunk(
                chunk_id=chunk_id,
                source=meta.get("source", ""),
                text=doc,
                metadata=meta,
            ))
        return chunks

    def retrieve(
        self,
        question: str,
        top_k: int = TOP_K_DOCS,
        use_hyde: bool = False,
        hyde_client=None,
    ) -> str:
        """Return retrieved passages joined as a single string for prompt injection."""
        chunks = self.retrieve_chunks(question, top_k, use_hyde, hyde_client)
        if not chunks:
            return ""
        return "\n\n---\n\n".join(
            f"[Source: {Path(c.source).name}]\n{c.text}" for c in chunks
        )

    def is_indexed(self, folder_path: str) -> bool:
        """Return True if any document from folder_path has been indexed."""
        try:
            collection = self._get_collection()
            folder_name = str(Path(folder_path))
            results = collection.get(
                where={"folder": folder_name},
                limit=1,
            )
            return len(results["ids"]) > 0
        except Exception:
            return False

    # ── Indexing internals ────────────────────────────────────────────────────

    def _index_file(self, file_path: Path) -> int:
        """Load, chunk, and upsert a single file. Returns chunk count."""
        text = self._load_file(file_path)
        if not text.strip():
            logger.warning("Empty text extracted from %s", file_path.name)
            return 0

        chunks = self._chunk_text(text)
        collection = self._get_collection()

        ids, documents, metadatas = [], [], []
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{file_path.name}::{i}"
            ids.append(chunk_id)
            documents.append(chunk_text)
            metadatas.append({
                "source": str(file_path),
                "filename": file_path.name,
                "chunk_index": i,
                "file_type": file_path.suffix.lower().lstrip("."),
                "folder": str(file_path.parent),
            })

        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        return len(chunks)

    def _load_file(self, file_path: Path) -> str:
        """Extract raw text from a PDF, DOCX, or TXT file."""
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            return self._load_pdf(file_path)
        elif suffix == ".docx":
            return self._load_docx(file_path)
        elif suffix == ".txt":
            return file_path.read_text(encoding="utf-8", errors="ignore")
        else:
            logger.warning("Unsupported file type: %s", suffix)
            return ""

    def _load_pdf(self, file_path: Path) -> str:
        try:
            import pypdf
            reader = pypdf.PdfReader(str(file_path))
            pages = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages.append(text.strip())
            return "\n\n".join(pages)
        except Exception as e:
            logger.error("Failed to load PDF %s: %s", file_path.name, e)
            return ""

    def _load_docx(self, file_path: Path) -> str:
        try:
            import docx
            doc = docx.Document(str(file_path))
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            return "\n\n".join(paragraphs)
        except Exception as e:
            logger.error("Failed to load DOCX %s: %s", file_path.name, e)
            return ""

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks of approximately chunk_size words.
        Uses word boundaries to avoid cutting mid-word.
        """
        words = text.split()
        if not words:
            return []

        step = self.chunk_size - self.chunk_overlap
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            if end == len(words):
                break
            start += step

        return chunks

    # ── HyDE ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _generate_hypothetical_answer(client, question: str) -> str:
        """
        Generate a hypothetical document passage that would answer the question.
        The passage is then embedded instead of the bare question (HyDE technique).
        Requires one Claude API call.
        """
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": (
                    f"Write a short passage (2-3 sentences) from a policy document "
                    f"that directly answers this question:\n\n{question}\n\n"
                    "Write only the passage, no preamble."
                ),
            }],
        )
        return response.content[0].text.strip()

    # ── ChromaDB ──────────────────────────────────────────────────────────────

    def _get_collection(self):
        if self._collection is not None:
            return self._collection

        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        ef = SentenceTransformerEmbeddingFunction(model_name=self.model_name)
        client = chromadb.PersistentClient(path=self.chroma_path)
        self._collection = client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=ef,
        )
        return self._collection
