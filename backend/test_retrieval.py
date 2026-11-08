"""Tests for semantic search over persisted Ollie document chunks."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag.retrieval import RetrievalError, search_documents
from rag.vector_store import VectorStore


class RetrievalTests(unittest.TestCase):
    """Verify search ranking, result metadata, input validation, and integration."""

    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_directory.cleanup)
        self.persist_directory = Path(self.temp_directory.name) / "chroma_db"

    @staticmethod
    def fake_document_embeddings(texts: list[str]) -> list[list[float]]:
        vectors = {
            "A bread recipe uses flour, yeast, and a hot oven.": [1.0, 0.0, 0.0],
            "A soccer match is played by two teams on a field.": [0.0, 1.0, 0.0],
            "The planets orbit the sun in our solar system.": [0.0, 0.0, 1.0],
        }
        return [vectors[text] for text in texts]

    @staticmethod
    def seed_store(persist_directory: Path) -> None:
        chunks = [
            {"text": "A bread recipe uses flour, yeast, and a hot oven.", "page_number": 1},
            {"text": "A soccer match is played by two teams on a field.", "page_number": 2},
            {"text": "The planets orbit the sun in our solar system.", "page_number": 3},
        ]
        with patch(
            "rag.vector_store.embed_texts",
            side_effect=RetrievalTests.fake_document_embeddings,
        ):
            with VectorStore(persist_directory) as store:
                store.add_chunks(chunks, "sample-document", "sample.pdf")

    def test_search_returns_relevant_chunks_with_metadata_and_distances(self) -> None:
        self.seed_store(self.persist_directory)

        with patch("rag.retrieval.embed_text", return_value=[0.99, 0.01, 0.0]) as embed_query:
            results = search_documents(
                "How can I bake bread?",
                persist_directory=self.persist_directory,
            )

        embed_query.assert_called_once_with("How can I bake bread?")
        self.assertEqual(len(results), 3)
        self.assertIn("bread recipe", results[0]["text"])
        self.assertEqual(
            results[0]["metadata"],
            {
                "document_id": "sample-document",
                "filename": "sample.pdf",
                "page_number": 1,
                "chunk_index": 0,
            },
        )
        self.assertEqual(results[0]["document_id"], "sample-document")
        self.assertIsInstance(results[0]["id"], str)
        self.assertIsInstance(results[0]["distance"], float)
        self.assertAlmostEqual(
            results[0]["similarity"],
            1.0 - results[0]["distance"],
        )

    def test_top_k_limits_the_number_of_results(self) -> None:
        self.seed_store(self.persist_directory)

        with patch("rag.retrieval.embed_text", return_value=[1.0, 0.0, 0.0]):
            results = search_documents(
                "What do I need to bake bread?",
                top_k=1,
                persist_directory=self.persist_directory,
            )

        self.assertEqual(len(results), 1)
        self.assertIn("bread recipe", results[0]["text"])

    def test_empty_collection_returns_no_matches_without_embedding(self) -> None:
        with VectorStore(self.persist_directory):
            pass

        with patch("rag.retrieval.embed_text") as embed_query:
            self.assertEqual(
                search_documents("Find a document", persist_directory=self.persist_directory),
                [],
            )
            embed_query.assert_not_called()

    def test_missing_collection_raises_retrieval_error(self) -> None:
        with self.assertRaisesRegex(RetrievalError, "was not found"):
            search_documents("Find a document", persist_directory=self.persist_directory)

    def test_invalid_queries_and_top_k_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            search_documents("")
        with self.assertRaises(ValueError):
            search_documents("   ")
        with self.assertRaises(TypeError):
            search_documents(None)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            search_documents("search", top_k=0)
        with self.assertRaises(TypeError):
            search_documents("search", top_k=True)

    def test_database_errors_are_reported(self) -> None:
        self.seed_store(self.persist_directory)
        with patch("rag.retrieval.embed_text", return_value=[1.0, 0.0, 0.0]):
            with patch(
                "rag.retrieval.chromadb.PersistentClient",
                side_effect=OSError("database unavailable"),
            ):
                with self.assertRaisesRegex(RetrievalError, "Could not search"):
                    search_documents("Search terms", persist_directory=self.persist_directory)

    def test_real_ollama_embedding_retrieval_integration(self) -> None:
        chunks = [
            {
                "text": (
                    "Bread dough rises when yeast ferments sugars, and the loaf is baked "
                    "in a hot oven."
                ),
                "page_number": 1,
            },
            {
                "text": "A goalkeeper defends the net during a soccer match.",
                "page_number": 2,
            },
        ]
        with VectorStore(self.persist_directory) as store:
            store.add_chunks(chunks, "integration-document", "integration.pdf")

        results = search_documents(
            "How do yeast and an oven help make bread?",
            persist_directory=self.persist_directory,
        )
        self.assertEqual(len(results), 2)
        self.assertIn("Bread dough rises", results[0]["text"])
        self.assertEqual(
            results[0]["metadata"]["document_id"],
            "integration-document",
        )


if __name__ == "__main__":
    unittest.main()
