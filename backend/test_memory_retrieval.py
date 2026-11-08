"""Tests for keyword-based long-term memory retrieval."""

import tempfile
import unittest
from pathlib import Path

from memory.memory_models import MemoryCategory, MemoryStatus
from memory.memory_service import MemoryService
from memory.memory_store import MemoryStore


class MemoryRetrievalTests(unittest.TestCase):
    """Verify relevance, importance, filtering, access tracking, and edge cases."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        store = MemoryStore(Path(self.temporary_directory.name) / "ollie.db")
        self.store = store
        self.service = MemoryService(store)

    def test_retrieves_relevant_memories_in_relevance_order(self) -> None:
        exact = self.service.add_memory(
            "I prefer Python for backend development.",
            category=MemoryCategory.PREFERENCES,
            importance=0.4,
        )
        partial = self.service.add_memory(
            "I am learning Python.",
            category=MemoryCategory.EDUCATION,
            importance=1.0,
        )
        self.service.add_memory(
            "I enjoy hiking on weekends.",
            category=MemoryCategory.PERSONAL,
            importance=0.9,
        )

        results = self.service.retrieve_memories("Python backend")

        self.assertEqual([memory.id for memory in results], [exact.id, partial.id])
        self.assertTrue(all(memory.status == MemoryStatus.ACTIVE for memory in results))

    def test_category_filtering(self) -> None:
        education = self.service.add_memory(
            "I study Python programming.",
            category=MemoryCategory.EDUCATION,
            importance=0.6,
        )
        self.service.add_memory(
            "I prefer Python tools at work.",
            category=MemoryCategory.PREFERENCES,
            importance=0.9,
        )

        results = self.service.retrieve_memories("Python", category="education")

        self.assertEqual([memory.id for memory in results], [education.id])

    def test_importance_ranks_equally_relevant_memories(self) -> None:
        lower = self.service.add_memory(
            "I use Python for development.",
            category=MemoryCategory.CAREER,
            importance=0.2,
        )
        higher = self.service.add_memory(
            "I use Python for development at work.",
            category=MemoryCategory.PREFERENCES,
            importance=0.95,
        )

        results = self.service.retrieve_memories("Python development")

        self.assertEqual([memory.id for memory in results], [higher.id, lower.id])

    def test_top_k_limits_results(self) -> None:
        memories = [
            self.service.add_memory(
                f"I use Python for project {index}.",
                category=MemoryCategory.PROJECTS,
                importance=0.4 + index * 0.1,
            )
            for index in range(4)
        ]

        results = self.service.retrieve_memories("Python project", top_k=2)

        self.assertEqual(len(results), 2)
        self.assertEqual([memory.id for memory in results], [memories[3].id, memories[2].id])

    def test_archived_memories_are_excluded(self) -> None:
        active = self.service.add_memory(
            "I prefer Python for scripting.",
            category=MemoryCategory.PREFERENCES,
        )
        archived = self.service.add_memory(
            "I prefer Python for automation.",
            category=MemoryCategory.PREFERENCES,
            importance=1.0,
        )
        self.service.archive_memory(archived.id)

        results = self.service.retrieve_memories("Python")

        self.assertEqual([memory.id for memory in results], [active.id])

    def test_access_timestamp_updates_only_for_returned_matches(self) -> None:
        matching = self.service.add_memory(
            "I prefer Python for scripting.",
            category=MemoryCategory.PREFERENCES,
        )
        unrelated = self.service.add_memory(
            "I enjoy gardening in spring.",
            category=MemoryCategory.PERSONAL,
        )
        self.assertIsNone(self.store.get_memory(matching.id, touch=False).last_accessed_at)
        self.assertIsNone(self.store.get_memory(unrelated.id, touch=False).last_accessed_at)

        results = self.service.retrieve_memories("Python")

        self.assertEqual([memory.id for memory in results], [matching.id])
        self.assertIsNotNone(results[0].last_accessed_at)
        self.assertIsNone(self.store.get_memory(unrelated.id, touch=False).last_accessed_at)

    def test_empty_database_and_no_match_return_empty_lists(self) -> None:
        self.assertEqual(self.service.retrieve_memories("Python"), [])
        self.service.add_memory(
            "I enjoy gardening in spring.",
            category=MemoryCategory.PERSONAL,
        )
        self.assertEqual(self.service.retrieve_memories("Python"), [])

    def test_duplicate_content_is_returned_once(self) -> None:
        self.service.add_memory(
            "I prefer Python for backend development.",
            category=MemoryCategory.PREFERENCES,
            importance=0.4,
        )
        self.service.add_memory(
            "I prefer Python for backend development.",
            category=MemoryCategory.OTHER,
            importance=0.9,
        )

        results = self.service.retrieve_memories("Python backend")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].category, MemoryCategory.OTHER)
        self.assertEqual(results[0].importance, 0.9)

    def test_invalid_queries_categories_and_top_k_are_rejected(self) -> None:
        with self.assertRaises(TypeError):
            self.service.retrieve_memories(None)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            self.service.retrieve_memories("")
        with self.assertRaises(ValueError):
            self.service.retrieve_memories("the and of")
        with self.assertRaises(ValueError):
            self.service.retrieve_memories("Python", category="invalid")
        with self.assertRaises(ValueError):
            self.service.retrieve_memories("Python", top_k=0)
        with self.assertRaises(TypeError):
            self.service.retrieve_memories("Python", top_k=True)


if __name__ == "__main__":
    unittest.main()
