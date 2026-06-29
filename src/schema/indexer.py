import logging
from pathlib import Path
from typing import Dict, List

from .serializer import SchemaChunk, SchemaSerializer
from src.config import (
    CHROMA_PATH,
    EMBEDDING_MODEL,
    SCHEMA_COLLECTION,
    SAMPLE_ROWS,
    MAX_COLUMNS,
)

logger = logging.getLogger(__name__)


class SchemaIndexer:
    """Embed SchemaChunks and persist them to a ChromaDB collection."""

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

        logger.info("Loading embedding model '%s' (downloads ~80 MB on first run)…", self.model_name)
        ef = SentenceTransformerEmbeddingFunction(model_name=self.model_name)

        client = chromadb.PersistentClient(path=self.chroma_path)
        self._collection = client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=ef,
        )
        return self._collection

    def index_database(self, chunks: List[SchemaChunk]) -> int:
        """Upsert chunks for one database. Safe to call multiple times (idempotent)."""
        if not chunks:
            logger.warning("index_database called with empty chunk list — skipping.")
            return 0

        collection = self._get_collection()
        collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )
        logger.info("Indexed %d tables for db_id='%s'", len(chunks), chunks[0].db_id)
        return len(chunks)

    def index_all_databases(
        self, db_root: str, show_progress: bool = True
    ) -> Dict[str, int]:
        """
        Walk db_root, serialize and index every SQLite file found.
        BIRD convention: dev_databases/{db_id}/{db_id}.sqlite
        Returns {db_id: chunk_count}.
        """
        serializer = SchemaSerializer(sample_rows=SAMPLE_ROWS, max_columns=MAX_COLUMNS)
        root = Path(db_root)
        sqlite_files = list(root.rglob("*.sqlite")) + list(root.rglob("*.db"))

        if not sqlite_files:
            logger.warning("No SQLite files found under %s", db_root)
            return {}

        iterator = sqlite_files
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(sqlite_files, desc="Indexing databases")
            except ImportError:
                pass

        results: Dict[str, int] = {}
        for sqlite_path in iterator:
            db_id = sqlite_path.parent.name
            chunks = serializer.serialize_database(str(sqlite_path), db_id)
            count = self.index_database(chunks)
            results[db_id] = count

        logger.info("Indexed %d databases total.", len(results))
        return results
