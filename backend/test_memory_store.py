"""Tests for the SQLite long-term memory layer."""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from memory.memory_models import MemoryCategory, MemoryStatus
from memory.memory_service import MemoryNotFoundError, MemoryService
from memory.memory_store import DuplicateMemoryError, MemoryStore


class MemoryStoreTests(unittest.TestCase):
    """Test memory persistence and ensure unrelated conversation rows are untouched."""

    @staticmethod
    def execute_sql(
        database_path: Path,
        query: str,
        parameters: tuple[object, ...] = (),
    ) -> list[tuple[object, ...]]:
        connection = sqlite3.connect(database_path)
        try:
            cursor = connection.execute(query, parameters)
            rows = cursor.fetchall()
            connection.commit()
            return rows
        finally:
            connection.close()

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "ollie.db"
        self.execute_sql(
            self.database_path,
            """
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """,
        )
        self.execute_sql(
            self.database_path,
            """
            INSERT INTO messages (conversation_id, role, content)
            VALUES (?, ?, ?)
            """,
            ("conversation-1", "user", "Remember my existing chat."),
        )
        self.store = MemoryStore(self.database_path)
        self.service = MemoryService(self.store)

    def test_initialization_creates_memory_table_and_indexes(self) -> None:
        tables = {
            row[0]
            for row in self.execute_sql(
                self.database_path,
                "SELECT name FROM sqlite_master WHERE type = 'table'",
            )
        }
        indexes = {
            row[0]
            for row in self.execute_sql(
                self.database_path,
                "SELECT name FROM sqlite_master WHERE type = 'index'",
            )
        }

        self.assertIn("messages", tables)
        self.assertIn("memories", tables)
        self.assertIn("idx_memories_category_status_importance", indexes)
        self.assertIn("idx_memories_source_conversation", indexes)

    def test_add_and_retrieve_memory(self) -> None:
        created = self.service.add_memory(
            "  I prefer concise explanations.  ",
            category="preferences",
            importance=0.8,
            source_conversation_id="conversation-1",
        )

        self.assertEqual(created.content, "I prefer concise explanations.")
        self.assertEqual(created.category, MemoryCategory.PREFERENCES)
        self.assertEqual(created.importance, 0.8)
        self.assertEqual(created.source_conversation_id, "conversation-1")
        self.assertEqual(created.status, MemoryStatus.ACTIVE)

        retrieved = self.service.get_memory(created.id)
        self.assertEqual(retrieved.id, created.id)
        self.assertIsNotNone(retrieved.last_accessed_at)

    def test_update_memory(self) -> None:
        created = self.service.add_memory(
            "I am learning Python.",
            category=MemoryCategory.EDUCATION,
            importance=0.4,
        )
        updated = self.service.update_memory(
            created.id,
            content="I am learning Python and SQLite.",
            importance=0.9,
        )

        self.assertEqual(updated.content, "I am learning Python and SQLite.")
        self.assertEqual(updated.importance, 0.9)
        self.assertEqual(updated.category, MemoryCategory.EDUCATION)
        self.assertGreaterEqual(updated.updated_at, created.updated_at)

    def test_filter_by_category(self) -> None:
        education = self.service.add_memory(
            "I am studying databases.",
            category=MemoryCategory.EDUCATION,
        )
        career = self.service.add_memory(
            "I work as a software developer.",
            category=MemoryCategory.CAREER,
        )

        results = self.service.get_memories(category="education")
        self.assertEqual([memory.id for memory in results], [education.id])
        self.assertNotIn(career.id, [memory.id for memory in results])

    def test_archive_and_delete_memory(self) -> None:
        archived = self.service.add_memory(
            "I was working on an old project.",
            category=MemoryCategory.PROJECTS,
        )
        deleted = self.service.add_memory(
            "Remove this temporary memory.",
            category=MemoryCategory.OTHER,
        )

        archived_record = self.service.archive_memory(archived.id)
        self.assertEqual(archived_record.status, MemoryStatus.ARCHIVED)
        self.assertEqual(
            self.service.get_memories(category=MemoryCategory.PROJECTS),
            [],
        )
        self.assertEqual(
            self.service.get_memories(
                category=MemoryCategory.PROJECTS,
                include_archived=True,
            )[0].id,
            archived.id,
        )

        self.service.delete_memory(deleted.id)
        with self.assertRaises(MemoryNotFoundError):
            self.service.get_memory(deleted.id)

    def test_duplicate_add_is_idempotent_and_store_rejects_duplicate(self) -> None:
        original = self.service.add_memory(
            "I prefer clear, concise answers.",
            category=MemoryCategory.PREFERENCES,
            importance=0.6,
        )
        duplicate = self.service.add_memory(
            "  i   prefer CLEAR, concise answers. ",
            category="preferences",
            importance=0.9,
        )
        self.assertEqual(duplicate.id, original.id)
        self.assertEqual(duplicate.importance, 0.6)

        different_category = self.service.add_memory(
            "I prefer clear, concise answers.",
            category=MemoryCategory.OTHER,
        )
        self.assertNotEqual(different_category.id, original.id)

        direct_duplicate = type(original)(
            id="different-id",
            content=original.content,
            category=original.category,
            importance=original.importance,
            source_conversation_id=None,
            created_at=original.created_at,
            updated_at=original.updated_at,
            last_accessed_at=None,
            status=MemoryStatus.ACTIVE,
        )
        with self.assertRaises(DuplicateMemoryError):
            self.store.create_memory(direct_duplicate)

    def test_persistence_after_reopening_database(self) -> None:
        created = self.service.add_memory(
            "Ollie should remember my current goal.",
            category=MemoryCategory.GOALS,
            importance=1.0,
            source_conversation_id="conversation-2",
        )

        reopened_store = MemoryStore(self.database_path)
        reopened_service = MemoryService(reopened_store)
        persisted = reopened_service.get_memory(created.id)
        self.assertEqual(persisted.content, created.content)
        self.assertEqual(persisted.category, MemoryCategory.GOALS)
        self.assertEqual(persisted.source_conversation_id, "conversation-2")

    def test_conversation_history_is_unchanged(self) -> None:
        self.service.add_memory(
            "I prefer local-first software.",
            category=MemoryCategory.PREFERENCES,
        )

        messages = self.execute_sql(
            self.database_path,
            "SELECT conversation_id, role, content FROM messages",
        )
        memory_count = self.execute_sql(
            self.database_path,
            "SELECT COUNT(*) FROM memories",
        )[0][0]

        self.assertEqual(
            messages,
            [("conversation-1", "user", "Remember my existing chat.")],
        )
        self.assertEqual(memory_count, 1)

    def test_invalid_memory_data_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.service.add_memory(" ", category="personal")
        with self.assertRaises(ValueError):
            self.service.add_memory("A fact", category="unknown")
        with self.assertRaises(ValueError):
            self.service.add_memory("A fact", importance=1.5)
        memory = self.service.add_memory("A valid fact")
        with self.assertRaises(ValueError):
            self.service.update_memory(memory.id, importance=1.2)
        with self.assertRaises(MemoryNotFoundError):
            self.service.update_memory("missing-id", importance=0.2)


if __name__ == "__main__":
    unittest.main()
