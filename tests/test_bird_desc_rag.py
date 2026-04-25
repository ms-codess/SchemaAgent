import csv
from pathlib import Path

from src.bird_desc_rag import BirdDescRAG


class FakeCollection:
    def __init__(self):
        self.upserts = []
        self.last_query = None
        self.indexed_dbs = set()

    def get(self, where=None, limit=None):
        db_id = None if where is None else where.get("db_id")
        if db_id not in self.indexed_dbs:
            return {"ids": []}
        return {"ids": ["db::table"]}

    def upsert(self, ids, documents, metadatas):
        self.upserts.append(
            {"ids": ids, "documents": documents, "metadatas": metadatas}
        )
        for metadata in metadatas:
            self.indexed_dbs.add(metadata["db_id"])

    def query(self, query_texts, n_results, where):
        self.last_query = {
            "query_texts": query_texts,
            "n_results": n_results,
            "where": where,
        }
        return {
            "documents": [[
                "Table: accounts\n  Column \"account_id\" [integer]\n    account identifier"
            ]],
            "metadatas": [[{"db_id": where["db_id"], "table": "accounts"}]],
        }


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "original_column_name",
                "column_name",
                "column_description",
                "data_format",
                "value_description",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_rows_to_text_includes_alias_format_and_values():
    text = BirdDescRAG._rows_to_text(
        "transactions",
        [
            {
                "original_column_name": "currency",
                "column_name": "payment currency",
                "column_description": "Currency used for the transaction",
                "data_format": "text",
                "value_description": "CZK = Czech koruna; EUR = euro",
            }
        ],
    )

    assert "Table: transactions" in text
    assert 'Column "currency" (payment currency) [text]' in text
    assert "Currency used for the transaction" in text
    assert "CZK = Czech koruna" in text


def test_index_db_upserts_one_document_per_csv(tmp_path):
    db_root = tmp_path / "dev_databases"
    desc_dir = db_root / "financial" / "database_description"
    desc_dir.mkdir(parents=True)

    write_csv(
        desc_dir / "accounts.csv",
        [
            {
                "original_column_name": "account_id",
                "column_name": "",
                "column_description": "account identifier",
                "data_format": "integer",
                "value_description": "",
            }
        ],
    )
    write_csv(
        desc_dir / "transactions.csv",
        [
            {
                "original_column_name": "currency",
                "column_name": "",
                "column_description": "payment currency",
                "data_format": "text",
                "value_description": "CZK, EUR",
            }
        ],
    )

    rag = BirdDescRAG()
    fake = FakeCollection()
    rag._collection = fake

    count = rag.index_db(str(db_root), "financial")

    assert count == 2
    assert len(fake.upserts) == 1
    assert fake.upserts[0]["ids"] == [
        "financial::accounts",
        "financial::transactions",
    ]


def test_index_db_returns_zero_when_description_folder_missing(tmp_path):
    rag = BirdDescRAG()
    rag._collection = FakeCollection()

    count = rag.index_db(str(tmp_path), "missing")

    assert count == 0


def test_retrieve_formats_dictionary_passages_for_one_db():
    rag = BirdDescRAG()
    fake = FakeCollection()
    fake.indexed_dbs.add("financial")
    rag._collection = fake

    text = rag.retrieve("Which account used EUR?", db_id="financial", top_k=3)

    assert "[Data dictionary - table 'accounts']" in text
    assert "account identifier" in text
    assert fake.last_query["where"] == {"db_id": "financial"}
    assert fake.last_query["n_results"] == 1
