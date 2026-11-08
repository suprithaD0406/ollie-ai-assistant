"""Tests for persistent ChromaDB chunk storage."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag.vector_store import (
    COLLECTION_NAME,
    DuplicateChunkError,
    VectorStore,
    VectorStoreError,
    add_chunks,
    get_chunk_count,
)


class VectorStoreTests(unittest.TestCase):
    """Exercise collection creation, chunk storage, validation, and persistence."""

    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_directory.cleanup)
        self.persist_directory = Path(self.temp_directory.name) / "chroma_db"

    @staticmethod
    def fake_embeddings(texts: list[str]) -> list[list[float]]:
        return [[float(index + 1), 0.25, 0.5] for index, _ in enumerate(texts)]

    def test_collection_initialization_uses_cosine_distance(self) -> None:
        with VectorStore(self.persist_directory) as store:
            self.assertEqual(store.collection.name, COLLECTION_NAME)
            self.assertEqual(store.collection.metadata["hnsw:space"], "cosine")
            self.assertEqual(store.get_chunk_count(), 0)

    def test_add_and_read_multiple_chunks(self) -> None:
        chunks = [
            {"text": "First page text.", "page_number": 1},
            {"text": "Second page text.", "page_number": 2},
        ]

        with patch("rag.vector_store.embed_texts", side_effect=self.fake_embeddings) as embed:
            with VectorStore(self.persist_directory) as store:
                ids = store.add_chunks(chunks, "document-1", "sample.pdf")

                self.assertEqual(embed.call_args.args[0], [chunk["text"] for chunk in chunks])
                self.assertEqual(len(ids), 2)
                self.assertEqual(len(set(ids)), 2)
                self.assertEqual(store.get_chunk_count(), 2)

                records = store.collection.get(ids=ids, include=["documents", "metadatas"])
                self.assertEqual(records["documents"], ["First page text.", "Second page text."])
                self.assertEqual(
                    records["metadatas"],
                    [
                        {
                            "document_id": "document-1",
                            "filename": "sample.pdf",
                            "page_number": 1,
                            "chunk_index": 0,
                        },
                        {
                            "document_id": "document-1",
                            "filename": "sample.pdf",
                            "page_number": 2,
                            "chunk_index": 1,
                        },
                    ],
                )

    def test_persistence_after_reinitializing_client(self) -> None:
        chunks = [
            {"text": "Persistent first chunk.", "page_number": 1},
            {"text": "Persistent second chunk.", "page_number": 1},
        ]

        with patch("rag.vector_store.embed_texts", side_effect=self.fake_embeddings):
            with VectorStore(self.persist_directory) as store:
                ids = store.add_chunks(chunks, "document-2", "persistent.pdf")

            with VectorStore(self.persist_directory) as reopened_store:
                self.assertEqual(reopened_store.get_chunk_count(), 2)
                records = reopened_store.collection.get(
                    ids=ids,
                    include=["documents", "metadatas"],
                )
                self.assertEqual(
                    records["documents"],
                    ["Persistent first chunk.", "Persistent second chunk."],
                )
                self.assertEqual(
                    [metadata["filename"] for metadata in records["metadatas"]],
                    ["persistent.pdf", "persistent.pdf"],
                )

    def test_duplicate_ids_are_rejected(self) -> None:
        chunk = [{"text": "One chunk.", "page_number": 1}]
        duplicate_uuid = type("FixedUUID", (), {"hex": "fixed-id"})()

        with patch("rag.vector_store.embed_texts", side_effect=self.fake_embeddings):
            with VectorStore(self.persist_directory) as store:
                with patch("rag.vector_store.uuid.uuid4", return_value=duplicate_uuid):
                    store.add_chunks(chunk, "document-3", "duplicate.pdf")
                    with self.assertRaises(DuplicateChunkError):
                        store.add_chunks(chunk, "document-3", "duplicate.pdf")
                self.assertEqual(store.get_chunk_count(), 1)

    def test_invalid_input_is_rejected_before_embedding(self) -> None:
        with patch("rag.vector_store.embed_texts") as embed:
            with VectorStore(self.persist_directory) as store:
                with self.assertRaises(ValueError):
                    store.add_chunks(
                        [{"text": "Invalid page.", "page_number": 0}],
                        "document-4",
                        "invalid.pdf",
                    )
                embed.assert_not_called()
        with self.assertRaises(ValueError):
            VectorStore("")

    def test_database_initialization_errors_are_reported(self) -> None:
        not_a_directory = self.persist_directory.parent / "not_a_directory"
        not_a_directory.write_text("not a directory", encoding="utf-8")
        with self.assertRaises(VectorStoreError):
            VectorStore(not_a_directory / "chroma_db")

    def test_module_level_add_and_count_helpers(self) -> None:
        chunks = [{"text": "Stored through helpers.", "page_number": 1}]
        with patch("rag.vector_store.embed_texts", side_effect=self.fake_embeddings):
            ids = add_chunks(
                chunks,
                "document-6",
                "helpers.pdf",
                self.persist_directory,
            )
        self.assertEqual(len(ids), 1)
        self.assertEqual(get_chunk_count(self.persist_directory), 1)

    def test_database_can_be_reopened_in_a_new_process(self) -> None:
        code = (
            "import json, sys; "
            "from rag.vector_store import VectorStore; "
            "store = VectorStore(sys.argv[1]); "
            "print(json.dumps({'count': store.get_chunk_count()})); "
            "store.close()"
        )
        with patch("rag.vector_store.embed_texts", side_effect=self.fake_embeddings):
            with VectorStore(self.persist_directory) as store:
                store.add_chunks(
                    [{"text": "Visible from another process.", "page_number": 1}],
                    "document-5",
                    "process.pdf",
                )

        result = subprocess.run(
            [sys.executable, "-c", code, str(self.persist_directory)],
            cwd=Path(__file__).parent,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(json.loads(result.stdout)["count"], 1)


if __name__ == "__main__":
    unittest.main()
