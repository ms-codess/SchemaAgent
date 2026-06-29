import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class SchemaChunk:
    chunk_id: str           # "{db_id}::{table_name}" — stable ChromaDB document ID
    db_id: str
    table_name: str
    text: str               # prose description injected into the SQL prompt
    metadata: dict = field(default_factory=dict)


class SchemaSerializer:
    """Convert a SQLite database into a list of SchemaChunks (one per table)."""

    def __init__(self, sample_rows: int = 3, max_columns: int = 20):
        self.sample_rows = sample_rows
        self.max_columns = max_columns

    def serialize_database(self, db_path: str, db_id: str) -> List[SchemaChunk]:
        """Return one SchemaChunk per table in the database."""
        path = Path(db_path)
        if not path.exists():
            return []

        try:
            uri = f"file:{path.as_posix()}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        except sqlite3.Error:
            return []

        try:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [
                row[0] for row in cursor.fetchall()
                if row[0] and not row[0].startswith("sqlite_")
            ]
            return [self._serialize_table(conn, db_id, t) for t in tables]
        finally:
            conn.close()

    def _serialize_table(
        self, conn: sqlite3.Connection, db_id: str, table_name: str
    ) -> SchemaChunk:
        cursor = conn.cursor()

        # Column names and declared types
        cursor.execute(f"PRAGMA table_info(\"{table_name}\")")
        columns = cursor.fetchall()  # (cid, name, type, notnull, dflt, pk)

        col_info = []
        for col in columns[: self.max_columns]:
            cid, name, col_type, *_ = col
            if not name or name.startswith("__"):
                continue
            samples = self._sample_values(conn, table_name, name)
            sample_str = f" — examples: {', '.join(samples)}" if samples else ""
            col_info.append(f"  - {name} ({col_type or 'TEXT'}){sample_str}")

        truncation_note = ""
        if len(columns) > self.max_columns:
            truncation_note = f"\n  ... ({len(columns) - self.max_columns} more columns)"

        text = (
            f"Table [{table_name}] in database [{db_id}].\n"
            f"Columns:\n"
            + "\n".join(col_info)
            + truncation_note
        )

        return SchemaChunk(
            chunk_id=f"{db_id}::{table_name}",
            db_id=db_id,
            table_name=table_name,
            text=text,
            metadata={
                "db_id": db_id,
                "table_name": table_name,
                "column_names": [c[1] for c in columns if c[1]],
                "column_types": [c[2] for c in columns if c[1]],
            },
        )

    def _sample_values(
        self, conn: sqlite3.Connection, table_name: str, column_name: str
    ) -> List[str]:
        try:
            cursor = conn.cursor()
            cursor.execute(
                f'SELECT DISTINCT "{column_name}" FROM "{table_name}" '
                f"WHERE \"{column_name}\" IS NOT NULL LIMIT {self.sample_rows}"
            )
            rows = cursor.fetchall()
            results = []
            for (val,) in rows:
                if isinstance(val, bytes):
                    continue  # skip binary blobs
                results.append(str(val)[:50])
            return results
        except sqlite3.Error:
            return []
