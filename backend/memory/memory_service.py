"""Business operations for creating and managing Ollie's long-term memories."""

from __future__ import annotations

import re

from memory.memory_models import (
    Memory,
    MemoryCategory,
    MemoryInput,
    normalize_memory_content,
    normalize_category,
    validate_content,
    validate_importance,
    validate_source_conversation_id,
)
from memory.memory_store import (
    DuplicateMemoryError,
    MemoryStore,
)
from memory.memory_updater import MemoryUpdateResult, MemoryUpdater

_QUERY_STOP_WORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "do", "for",
        "from", "how", "i", "in", "is", "it", "me", "my", "of", "on", "or",
        "please", "that", "the", "this", "to", "was", "what", "when", "where",
        "which", "who", "why", "with", "would", "you", "your",
    }
)


class MemoryNotFoundError(LookupError):
    """Raised when a requested memory does not exist."""


class MemoryService:
    """Validate memory operations and delegate persistence to MemoryStore."""

    def __init__(self, store: MemoryStore | None = None) -> None:
        """Create a service using the supplied store or the default Ollie database."""
        self.store = store if store is not None else MemoryStore()

    def add_memory(
        self,
        content: str,
        category: MemoryCategory | str = MemoryCategory.OTHER,
        importance: float = 0.5,
        source_conversation_id: str | None = None,
    ) -> Memory:
        """Add a memory, returning the existing active record for duplicates."""
        memory_input = MemoryInput(
            content=content,
            category=category,
            importance=importance,
            source_conversation_id=source_conversation_id,
        )
        memory = Memory.create(memory_input)
        try:
            return self.store.create_memory(memory)
        except DuplicateMemoryError:
            duplicate = self.store.get_active_duplicate(
                memory.category,
                memory.content,
            )
            if duplicate is None:
                raise
            return duplicate

    def get_memory(self, memory_id: str) -> Memory:
        """Retrieve one memory or raise MemoryNotFoundError."""
        memory = self.store.get_memory(memory_id)
        if memory is None:
            raise MemoryNotFoundError(f"Memory {memory_id!r} was not found.")
        return memory

    def get_memories(
        self,
        category: MemoryCategory | str | None = None,
        *,
        include_archived: bool = False,
        limit: int = 100,
    ) -> list[Memory]:
        """List active memories, optionally including archived records by category."""
        return self.store.list_memories(
            category,
            include_archived=include_archived,
            limit=limit,
        )

    def retrieve_memories(
        self,
        query: str,
        *,
        category: MemoryCategory | str | None = None,
        top_k: int = 5,
    ) -> list[Memory]:
        """Find active memories by keyword overlap, then rank by relevance and importance.

        Query-term coverage is the primary ranking signal; importance breaks ties
        between equally relevant memories. Only returned records have their
        ``last_accessed_at`` timestamp updated.
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string.")
        if not query.strip():
            raise ValueError("query cannot be empty.")
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be a positive integer.")
        if top_k < 1:
            raise ValueError("top_k must be a positive integer.")
        normalized_category = normalize_category(category) if category is not None else None

        query_terms = {
            token
            for token in re.findall(r"\w+", query.casefold(), flags=re.UNICODE)
            if token not in _QUERY_STOP_WORDS
        }
        if not query_terms:
            raise ValueError("query must contain at least one searchable keyword.")

        candidates = self.store.list_memories(
            normalized_category,
            include_archived=False,
            limit=None,
            touch=False,
        )
        ranked: list[tuple[tuple[float, int, float, str, str], Memory]] = []

        for memory in candidates:
            memory_terms = set(
                re.findall(r"\w+", memory.content.casefold(), flags=re.UNICODE)
            )
            matched_terms = query_terms & memory_terms
            if not matched_terms:
                continue

            relevance = len(matched_terms) / len(query_terms)
            rank = (
                relevance,
                len(matched_terms),
                memory.importance,
                memory.updated_at,
                memory.id,
            )
            ranked.append((rank, memory))

        ranked.sort(key=lambda result: result[0], reverse=True)
        selected: list[Memory] = []
        seen_content: set[str] = set()
        for _, memory in ranked:
            normalized_content = normalize_memory_content(memory.content)
            if normalized_content in seen_content:
                continue
            selected.append(memory)
            seen_content.add(normalized_content)
            if len(selected) == top_k:
                break

        if not selected:
            return []
        return self.store.touch_memories([memory.id for memory in selected])

    def update_memory(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        category: MemoryCategory | str | None = None,
        importance: float | None = None,
        source_conversation_id: str | None = None,
        restore_archived: bool = False,
    ) -> Memory:
        """Update an existing memory after validating the supplied values."""
        if content is not None:
            content = validate_content(content)
        if category is not None:
            category = MemoryInput(content="validation", category=category).category
        if importance is not None:
            importance = validate_importance(importance)
        if source_conversation_id is not None:
            source_conversation_id = validate_source_conversation_id(
                source_conversation_id
            )

        memory = self.store.update_memory(
            memory_id,
            content=content,
            category=category,
            importance=importance,
            source_conversation_id=source_conversation_id,
            restore_archived=restore_archived,
        )
        if memory is None:
            raise MemoryNotFoundError(f"Memory {memory_id!r} was not found.")
        return memory

    def apply_memory(
        self,
        memory_input: MemoryInput,
        *,
        explicit_correction: bool = False,
    ) -> MemoryUpdateResult:
        """Add, deduplicate, update, or flag a memory candidate for review."""
        return MemoryUpdater(self.store).apply(
            memory_input,
            explicit_correction=explicit_correction,
        )

    def archive_memory(self, memory_id: str) -> Memory:
        """Archive an existing memory."""
        memory = self.store.archive_memory(memory_id)
        if memory is None:
            raise MemoryNotFoundError(f"Memory {memory_id!r} was not found.")
        return memory

    def delete_memory(self, memory_id: str) -> None:
        """Permanently delete an existing memory."""
        if not self.store.delete_memory(memory_id):
            raise MemoryNotFoundError(f"Memory {memory_id!r} was not found.")
