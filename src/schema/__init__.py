from .serializer import SchemaSerializer, SchemaChunk
from .indexer import SchemaIndexer
from .retriever import SchemaRetriever
from src.config import TOP_K, CHROMA_PATH, SCHEMA_COLLECTION, EMBEDDING_MODEL


class SchemaRAG:
    """
    Facade over SchemaSerializer + SchemaIndexer + SchemaRetriever.
    This is the only class agent.py needs to import.
    """

    def __init__(
        self,
        chroma_path: str = CHROMA_PATH,
        collection_name: str = SCHEMA_COLLECTION,
        model_name: str = EMBEDDING_MODEL,
        top_k: int = TOP_K,
    ):
        self.top_k = top_k
        self._indexer = SchemaIndexer(chroma_path, collection_name, model_name)
        self._retriever = SchemaRetriever(chroma_path, collection_name, model_name)

    def index(self, db_path: str, db_id: str) -> int:
        from .serializer import SchemaSerializer
        from src.config import SAMPLE_ROWS, MAX_COLUMNS
        serializer = SchemaSerializer(sample_rows=SAMPLE_ROWS, max_columns=MAX_COLUMNS)
        chunks = serializer.serialize_database(db_path, db_id)
        return self._indexer.index_database(chunks)

    def index_all(self, db_root: str, show_progress: bool = True):
        return self._indexer.index_all_databases(db_root, show_progress)

    def get_schema_context(
        self, question: str, db_id: str, use_rag: bool = True
    ) -> str:
        """
        use_rag=True  → selective retrieval (top-k tables).
        use_rag=False → returns "" so caller falls back to full schema dump.
        """
        if not use_rag:
            return ""
        return self._retriever.retrieve_as_text(question, db_id, self.top_k)

    def is_indexed(self, db_id: str) -> bool:
        return self._retriever.is_indexed(db_id)


__all__ = [
    "SchemaRAG",
    "SchemaSerializer",
    "SchemaChunk",
    "SchemaIndexer",
    "SchemaRetriever",
]
