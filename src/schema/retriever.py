import logging
from typing import List

from .serializer import SchemaChunk
from src.config import (
    CHROMA_PATH,
    EMBEDDING_MODEL,
    SCHEMA_COLLECTION,
    TOP_K,
)

logger = logging.getLogger(__name__)


class SchemaRetriever:
    """Retrieve the most relevant table descriptions for a natural language question."""

    def __init__(
        self,
        chroma_path: str = CHROMA_PATH,
        collection_name: str = SCHEMA_COLLECTION,
        model_name: str = EMBEDDING_MODEL,
    ):
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        self.model_name = model_name
        self._collection = None

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

    def is_indexed(self, db_id: str) -> bool:
        """Return True if db_id has at least one document in the collection."""
        try:
            collection = self._get_collection()
            results = collection.get(where={"db_id": db_id}, limit=1)
            return len(results["ids"]) > 0
        except Exception:
            return False

    def retrieve(
        self, question: str, db_id: str, top_k: int = TOP_K
    ) -> List[SchemaChunk]:
        """Return the top-k most relevant SchemaChunks for this question + database."""
        if not self.is_indexed(db_id):
            logger.warning(
                "db_id='%s' has no indexed tables. Run SchemaIndexer first.", db_id
            )
            return []

        collection = self._get_collection()
        results = collection.query(
            query_texts=[question],
            n_results=top_k,
            where={"db_id": db_id},
        )

        chunks = []
        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]

        for chunk_id, doc, meta in zip(ids, docs, metas):
            chunks.append(
                SchemaChunk(
                    chunk_id=chunk_id,
                    db_id=meta.get("db_id", db_id),
                    table_name=meta.get("table_name", ""),
                    text=doc,
                    metadata=meta,
                )
            )
        return chunks

    def retrieve_as_text(
        self, question: str, db_id: str, top_k: int = TOP_K
    ) -> str:
        """Return retrieved table descriptions joined as a single string for prompt injection."""
        chunks = self.retrieve(question, db_id, top_k)
        return "\n\n".join(c.text for c in chunks)
