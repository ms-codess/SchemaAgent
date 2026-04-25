"""
BirdDescRAG - retrieval over BIRD database_description/*.csv files.

Each BIRD database ships with a database_description/ folder containing one
CSV per table. Each row describes a column: its human-readable name, full
description, data format, and valid values.

This class indexes those CSVs into ChromaDB and retrieves the most relevant
table descriptions for a given natural-language question, filtered to a
specific database.
"""

import csv
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

BIRD_DESC_COLLECTION = "bird_desc_chunks"
TOP_K_DESC = 5


class BirdDescRAG:
    """Index and retrieve BIRD column-description CSV files."""

    def __init__(
        self,
        chroma_path: str = "chroma_db",
        collection_name: str = BIRD_DESC_COLLECTION,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        self.model_name = model_name
        self._collection = None

    def is_indexed(self, db_id: str) -> bool:
        """Return True if a database already has indexed descriptions."""
        try:
            results = self._get_collection().get(where={"db_id": db_id}, limit=1)
            return len(results["ids"]) > 0
        except Exception:
            return False

    def index_db(self, db_root: str, db_id: str) -> int:
        """
        Index all database_description/*.csv files for one BIRD database.

        Returns the number of table-level chunks indexed, one per CSV.
        """
        if self.is_indexed(db_id):
            logger.debug("BirdDescRAG: %s already indexed", db_id)
            return 0

        desc_folder = Path(db_root) / db_id / "database_description"
        if not desc_folder.exists():
            logger.warning("No database_description folder for %s", db_id)
            return 0

        ids = []
        documents = []
        metadatas = []

        for csv_path in sorted(desc_folder.glob("*.csv")):
            rows = self._load_csv(csv_path)
            if not rows:
                continue

            table_name = csv_path.stem
            ids.append(f"{db_id}::{table_name}")
            documents.append(self._rows_to_text(table_name, rows))
            metadatas.append(
                {
                    "db_id": db_id,
                    "table": table_name,
                    "source": str(csv_path),
                }
            )

        if ids:
            self._get_collection().upsert(
                ids=ids, documents=documents, metadatas=metadatas
            )
            logger.info("BirdDescRAG: indexed %d tables for %s", len(ids), db_id)

        return len(ids)

    def retrieve(self, question: str, db_id: str, top_k: int = TOP_K_DESC) -> str:
        """
        Retrieve the most relevant table descriptions for a question.

        Returns a formatted string suitable for the SQL evidence field.
        """
        collection = self._get_collection()
        all_for_db = collection.get(where={"db_id": db_id})
        n_available = len(all_for_db["ids"])
        if n_available == 0:
            return ""

        results = collection.query(
            query_texts=[question],
            n_results=min(top_k, n_available),
            where={"db_id": db_id},
        )

        passages = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            passages.append(f"[Data dictionary - table '{meta['table']}']\n{doc}")
        return "\n\n".join(passages)

    def _load_csv(self, csv_path: Path) -> List[dict]:
        try:
            with open(csv_path, encoding="utf-8-sig", errors="ignore") as handle:
                return list(csv.DictReader(handle))
        except Exception as exc:
            logger.error("BirdDescRAG: failed to read %s: %s", csv_path.name, exc)
            return []

    @staticmethod
    def _rows_to_text(table_name: str, rows: List[dict]) -> str:
        """Convert CSV rows into a human-readable text block for embedding."""
        lines = [f"Table: {table_name}"]
        for row in rows:
            original = (row.get("original_column_name") or "").strip()
            alias = (row.get("column_name") or "").strip()
            description = (row.get("column_description") or "").strip()
            data_format = (row.get("data_format") or "").strip()
            values = (row.get("value_description") or "").strip()

            if not original:
                continue

            label = f'  Column "{original}"'
            if alias and alias != original:
                label += f" ({alias})"
            if data_format:
                label += f" [{data_format}]"

            parts = [label]
            if description:
                parts.append(f"    {description}")
            if values:
                parts.append(f"    Values: {values}")

            lines.append("\n".join(parts))

        return "\n".join(lines)

    def _get_collection(self):
        if self._collection is not None:
            return self._collection

        import chromadb
        from chromadb.utils.embedding_functions import (
            SentenceTransformerEmbeddingFunction,
        )

        embedding_function = SentenceTransformerEmbeddingFunction(
            model_name=self.model_name
        )
        client = chromadb.PersistentClient(path=self.chroma_path)
        self._collection = client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=embedding_function,
        )
        return self._collection
