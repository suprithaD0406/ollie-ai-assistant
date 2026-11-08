"""SQLite persistence layer for Ollie's long-term memories."""

from contextlib import contextmanager
from pathlib import Path
import sqlite3
from typing import Any, Generator

from memory.memory_models import (
    Memory,
    MemoryCategory,
    MemoryStatus,
    normalize_category,
    normalize_memory_content,
    validate_content,
    validate_importance,
    validate_source_conversation_id,
    utc_timestamp,
)


DEFAULT_DATABASE_PATH = Path(__file__).parent.parent / "ollie.db"


class MemoryStoreError(RuntimeError):
    """Raised when a memory database operation fails."""


class DuplicateMemoryError(MemoryStoreError):
    """Raised when an active memory duplicates an existing memory."""


class MemoryStore:
    """Perform parameterized SQLite operations for the memories table."""

    def __init__(self, database_path: str | Path | None = None) -> None:
        """Initialize the memories table in the existing Ollie SQLite database."""
        if database_path is not None and not isinstance(database_path, (str, Path)):
            raise TypeError("database_path must be a string or Path.")
        if isinstance(database_path, str) and not database_path.strip():
            raise ValueError("database_path cannot be empty.")
        self.database_path = Path(
            DEFAULT_DATABASE_PATH if database_path is None else database_path
        )
        try:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            self.initialize_database()
        except sqlite3.Error as exc:
            raise MemoryStoreError(f"Could not initialize memory database: {exc}") from exc
        except OSError as exc:
            raise MemoryStoreError(f"Could not access memory database path: {exc}") from exc

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(self.database_path, timeout=10.0)
            connection.row_factory = sqlite3.Row
            with connection:
                yield connection
        except sqlite3.Error as exc:
            raise MemoryStoreError(f"Memory database operation failed: {exc}") from exc
        finally:
            if connection is not None:
                connection.close()

    def initialize_database(self) -> None:
        """Create the memory table and indexes without touching other tables."""
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    normalized_content TEXT NOT NULL,
                    category TEXT NOT NULL CHECK (
                        category IN (
                            'personal', 'education', 'career', 'preferences',
                            'projects', 'goals', 'other'
                        )
                    ),
                    importance REAL NOT NULL CHECK (
                        importance >= 0.0 AND importance <= 1.0
                    ),
                    source_conversation_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_accessed_at TEXT,
                    status TEXT NOT NULL CHECK (status IN ('active', 'archived'))
                )
                """
            )
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_active_duplicate
                ON memories(category, normalized_content)
                WHERE status = 'active'
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_category_status_importance
                ON memories(category, status, importance DESC, updated_at DESC)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_source_conversation
                ON memories(source_conversation_id)
                """
            )

    def create_memory(self, memory: Memory) -> Memory:
        """Insert a new memory, rejecting an active duplicate."""
        if not isinstance(memory, Memory):
            raise TypeError("memory must be a Memory instance.")
        try:
            with self._connection() as connection:
                connection.execute(
                    """
                    INSERT INTO memories (
                        id, content, normalized_content, category, importance,
                        source_conversation_id, created_at, updated_at,
                        last_accessed_at, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        memory.id,
                        memory.content,
                        normalize_memory_content(memory.content),
                        memory.category.value,
                        memory.importance,
                        memory.source_conversation_id,
                        memory.created_at,
                        memory.updated_at,
                        memory.last_accessed_at,
                        memory.status.value,
                    ),
                )
        except MemoryStoreError as exc:
            if self._is_unique_violation(exc):
                raise DuplicateMemoryError("An active duplicate memory already exists.") from exc
            raise
        return memory

    def get_active_duplicate(
        self,
        category: MemoryCategory | str,
        content: str,
    ) -> Memory | None:
        """Find an active memory with matching category and normalized content."""
        normalized_category = normalize_category(category)
        normalized_content = normalize_memory_content(validate_content(content))
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM memories
                WHERE category = ? AND normalized_content = ? AND status = 'active'
                """,
                (normalized_category.value, normalized_content),
            ).fetchone()
        return Memory.from_row(dict(row)) if row is not None else None

    def get_active_memory_by_content(self, content: str) -> Memory | None:
        """Find an active memory with matching normalized content in any category."""
        normalized_content = normalize_memory_content(validate_content(content))
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM memories
                WHERE normalized_content = ? AND status = 'active'
                ORDER BY importance DESC, updated_at DESC, id
                LIMIT 1
                """,
                (normalized_content,),
            ).fetchone()
        return Memory.from_row(dict(row)) if row is not None else None

    def get_memory(self, memory_id: str, *, touch: bool = True) -> Memory | None:
        """Retrieve a memory by ID, updating its last-accessed timestamp by default."""
        memory_id = self._validate_memory_id(memory_id)
        with self._connection() as connection:
            if touch:
                connection.execute(
                    "UPDATE memories SET last_accessed_at = ? WHERE id = ?",
                    (utc_timestamp(), memory_id),
                )
            row = connection.execute(
                "SELECT * FROM memories WHERE id = ?",
                (memory_id,),
            ).fetchone()
        return Memory.from_row(dict(row)) if row is not None else None

    def list_memories(
        self,
        category: MemoryCategory | str | None = None,
        *,
        include_archived: bool = False,
        limit: int | None = 100,
        touch: bool = True,
    ) -> list[Memory]:
        """List memories, optionally filtered by category and lifecycle status."""
        if not isinstance(include_archived, bool):
            raise TypeError("include_archived must be a boolean.")
        if not isinstance(touch, bool):
            raise TypeError("touch must be a boolean.")
        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, int):
                raise TypeError("limit must be a positive integer or None.")
            if limit < 1:
                raise ValueError("limit must be a positive integer.")

        conditions: list[str] = []
        parameters: list[Any] = []
        if category is not None:
            conditions.append("category = ?")
            parameters.append(normalize_category(category).value)
        if not include_archived:
            conditions.append("status = ?")
            parameters.append(MemoryStatus.ACTIVE.value)
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        limit_clause = ""
        if limit is not None:
            limit_clause = "LIMIT ?"
            parameters.append(limit)

        with self._connection() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM memories
                {where_clause}
                ORDER BY importance DESC, updated_at DESC, id
                {limit_clause}
                """,
                parameters,
            ).fetchall()
            if rows and touch:
                accessed_at = utc_timestamp()
                connection.executemany(
                    "UPDATE memories SET last_accessed_at = ? WHERE id = ?",
                    [(accessed_at, row["id"]) for row in rows],
                )
                rows = [
                    {**dict(row), "last_accessed_at": accessed_at}
                    for row in rows
                ]

        return [Memory.from_row(dict(row)) for row in rows]

    def touch_memories(self, memory_ids: list[str]) -> list[Memory]:
        """Update access timestamps and return active memories in the supplied order."""
        if not isinstance(memory_ids, list):
            raise TypeError("memory_ids must be a list of IDs.")
        ids = list(dict.fromkeys(self._validate_memory_id(item) for item in memory_ids))
        if not ids:
            return []

        placeholders = ", ".join("?" for _ in ids)
        accessed_at = utc_timestamp()
        with self._connection() as connection:
            connection.executemany(
                f"""
                UPDATE memories SET last_accessed_at = ?
                WHERE id = ? AND status = ?
                """,
                [
                    (accessed_at, memory_id, MemoryStatus.ACTIVE.value)
                    for memory_id in ids
                ],
            )
            rows = connection.execute(
                f"""
                SELECT * FROM memories
                WHERE id IN ({placeholders}) AND status = ?
                """,
                (*ids, MemoryStatus.ACTIVE.value),
            ).fetchall()

        records = {row["id"]: Memory.from_row(dict(row)) for row in rows}
        return [records[memory_id] for memory_id in ids if memory_id in records]

    def update_memory(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        category: MemoryCategory | str | None = None,
        importance: float | None = None,
        source_conversation_id: str | None = None,
        restore_archived: bool = False,
    ) -> Memory | None:
        """Update an active memory, optionally restoring an archived one."""
        memory_id = self._validate_memory_id(memory_id)
        if not isinstance(restore_archived, bool):
            raise TypeError("restore_archived must be a boolean.")
        updates: dict[str, Any] = {}
        if content is not None:
            content = validate_content(content)
            updates["content"] = content
            updates["normalized_content"] = normalize_memory_content(content)
        if category is not None:
            updates["category"] = normalize_category(category).value
        if importance is not None:
            updates["importance"] = validate_importance(importance)
        if source_conversation_id is not None:
            updates["source_conversation_id"] = validate_source_conversation_id(
                source_conversation_id
            )
        if restore_archived:
            updates["status"] = MemoryStatus.ACTIVE.value
        if not updates:
            raise ValueError("At least one field must be provided to update.")

        updates["updated_at"] = utc_timestamp()

        assignments = ", ".join(f"{column} = ?" for column in updates)
        parameters: list[Any] = list(updates.values()) + [memory_id]
        status_clause = ""
        if not restore_archived:
            status_clause = " AND status = ?"
            parameters.append(MemoryStatus.ACTIVE.value)
        try:
            with self._connection() as connection:
                cursor = connection.execute(
                    f"UPDATE memories SET {assignments} WHERE id = ?{status_clause}",
                    parameters,
                )
                if cursor.rowcount == 0:
                    return None
                row = connection.execute(
                    "SELECT * FROM memories WHERE id = ?",
                    (memory_id,),
                ).fetchone()
        except MemoryStoreError as exc:
            if self._is_unique_violation(exc):
                raise DuplicateMemoryError(
                    "The update would duplicate another active memory."
                ) from exc
            raise
        return Memory.from_row(dict(row)) if row is not None else None

    def archive_memory(self, memory_id: str) -> Memory | None:
        """Mark a memory as archived and return its updated record."""
        memory_id = self._validate_memory_id(memory_id)
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE memories
                SET status = ?, updated_at = ?
                WHERE id = ?
                """,
                (MemoryStatus.ARCHIVED.value, utc_timestamp(), memory_id),
            )
            if cursor.rowcount == 0:
                return None
            row = connection.execute(
                "SELECT * FROM memories WHERE id = ?",
                (memory_id,),
            ).fetchone()
        return Memory.from_row(dict(row)) if row is not None else None

    def delete_memory(self, memory_id: str) -> bool:
        """Permanently delete a memory and return whether it existed."""
        memory_id = self._validate_memory_id(memory_id)
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM memories WHERE id = ?",
                (memory_id,),
            )
        return cursor.rowcount > 0

    @staticmethod
    def _validate_memory_id(memory_id: str) -> str:
        if not isinstance(memory_id, str):
            raise TypeError("memory_id must be a string.")
        memory_id = memory_id.strip()
        if not memory_id:
            raise ValueError("memory_id cannot be blank.")
        return memory_id

    @staticmethod
    def _is_unique_violation(error: MemoryStoreError) -> bool:
        cause = error.__cause__
        return isinstance(cause, sqlite3.IntegrityError) and "UNIQUE constraint failed" in str(
            cause
        )
