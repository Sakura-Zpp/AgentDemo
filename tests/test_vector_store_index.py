from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from langchain_core.documents import Document

from rag.vector_store import VectorStore


class Bm25PersistenceTest(unittest.TestCase):
    def test_bm25_index_round_trips_as_json(self) -> None:
        with TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
            index_path = Path(temporary_directory) / "bm25_index.json"
            writer = VectorStore.__new__(VectorStore)
            writer.bm25_persist_path = index_path
            writer.bm25_documents = [
                Document(page_content="可信文档", metadata={"source": "policy.txt"})
            ]
            writer._save_bm25_index()

            payload = json.loads(index_path.read_text(encoding="utf-8"))
            reader = VectorStore.__new__(VectorStore)
            reader.bm25_persist_path = index_path
            reader.bm25_documents = []
            reader._load_bm25_index()

            self.assertEqual(payload[0]["page_content"], "可信文档")
            self.assertEqual(reader.bm25_documents[0].metadata["source"], "policy.txt")

    def test_invalid_json_is_not_deserialized_or_executed(self) -> None:
        with TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
            index_path = Path(temporary_directory) / "bm25_index.json"
            index_path.write_text("not-json", encoding="utf-8")
            reader = VectorStore.__new__(VectorStore)
            reader.bm25_persist_path = index_path
            reader.bm25_documents = []

            reader._load_bm25_index()

            self.assertEqual(reader.bm25_documents, [])


if __name__ == "__main__":
    unittest.main()
