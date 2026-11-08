"""Validated data models for Ollie's long-term memories."""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import math
import uuid


class MemoryCategory(str, Enum):
    """Supported categories for a stored memory."""

    PERSONAL = "personal"
    EDUCATION = "education"
    CAREER = "career"
    PREFERENCES = "preferences"
    PROJECTS = "projects"
    GOALS = "goals"
    OTHER = "other"


class MemoryStatus(str, Enum):
    """Lifecycle state of a memory."""

    ACTIVE = "active"
    ARCHIVED = "archived"


def utc_timestamp() -> str:
    """Return a timezone-aware UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def normalize_category(category: MemoryCategory | str) -> MemoryCategory:
    """Validate and normalize a memory category."""
    if isinstance(category, MemoryCategory):
        return category
    if not isinstance(category, str):
        raise TypeError("category must be a string or MemoryCategory.")
    try:
        return MemoryCategory(category.strip().lower())
    except ValueError as exc:
        allowed = ", ".join(item.value for item in MemoryCategory)
        raise ValueError(f"category must be one of: {allowed}.") from exc


def normalize_memory_content(content: str) -> str:
    """Normalize whitespace and case for duplicate detection."""
    return " ".join(content.split()).casefold()


def validate_content(content: str) -> str:
    """Validate memory content and return its trimmed form."""
    if not isinstance(content, str):
        raise TypeError("content must be a string.")
    content = content.strip()
    if not content:
        raise ValueError("content cannot be empty.")
    return content


def validate_importance(importance: float) -> float:
    """Validate an importance score in the inclusive range 0.0 to 1.0."""
    if isinstance(importance, bool) or not isinstance(importance, (int, float)):
        raise TypeError("importance must be a number between 0.0 and 1.0.")
    score = float(importance)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("importance must be a finite number between 0.0 and 1.0.")
    return score


def validate_source_conversation_id(source_conversation_id: str | None) -> str | None:
    """Validate an optional source conversation identifier."""
    if source_conversation_id is None:
        return None
    if not isinstance(source_conversation_id, str):
        raise TypeError("source_conversation_id must be a string or None.")
    source_conversation_id = source_conversation_id.strip()
    if not source_conversation_id:
        raise ValueError("source_conversation_id cannot be blank.")
    return source_conversation_id


@dataclass(frozen=True, slots=True)
class MemoryInput:
    """Validated values required to create a memory."""

    content: str
    category: MemoryCategory | str = MemoryCategory.OTHER
    importance: float = 0.5
    source_conversation_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "content", validate_content(self.content))
        object.__setattr__(self, "category", normalize_category(self.category))
        object.__setattr__(self, "importance", validate_importance(self.importance))
        object.__setattr__(
            self,
            "source_conversation_id",
            validate_source_conversation_id(self.source_conversation_id),
        )


@dataclass(frozen=True, slots=True)
class Memory:
    """A persisted memory and its lifecycle metadata."""

    id: str
    content: str
    category: MemoryCategory
    importance: float
    source_conversation_id: str | None
    created_at: str
    updated_at: str
    last_accessed_at: str | None
    status: MemoryStatus

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("id must be a non-empty string.")
        object.__setattr__(self, "content", validate_content(self.content))
        object.__setattr__(self, "category", normalize_category(self.category))
        object.__setattr__(self, "importance", validate_importance(self.importance))
        object.__setattr__(
            self,
            "source_conversation_id",
            validate_source_conversation_id(self.source_conversation_id),
        )
        for field_name in ("created_at", "updated_at"):
            timestamp = getattr(self, field_name)
            if not isinstance(timestamp, str) or not timestamp.strip():
                raise ValueError(f"{field_name} must be a non-empty string.")
        if self.last_accessed_at is not None and (
            not isinstance(self.last_accessed_at, str) or not self.last_accessed_at.strip()
        ):
            raise ValueError("last_accessed_at must be a non-empty string or None.")
        if not isinstance(self.status, MemoryStatus):
            try:
                object.__setattr__(self, "status", MemoryStatus(self.status))
            except (TypeError, ValueError) as exc:
                raise ValueError("status must be 'active' or 'archived'.") from exc

    @classmethod
    def create(cls, memory_input: MemoryInput) -> "Memory":
        """Create a new active memory with a unique ID and UTC timestamps."""
        now = utc_timestamp()
        return cls(
            id=uuid.uuid4().hex,
            content=memory_input.content,
            category=normalize_category(memory_input.category),
            importance=memory_input.importance,
            source_conversation_id=memory_input.source_conversation_id,
            created_at=now,
            updated_at=now,
            last_accessed_at=None,
            status=MemoryStatus.ACTIVE,
        )

    @classmethod
    def from_row(cls, row: dict[str, object]) -> "Memory":
        """Construct a validated model from a SQLite row mapping."""
        return cls(
            id=str(row["id"]),
            content=str(row["content"]),
            category=normalize_category(str(row["category"])),
            importance=validate_importance(float(row["importance"])),
            source_conversation_id=(
                str(row["source_conversation_id"])
                if row["source_conversation_id"] is not None
                else None
            ),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            last_accessed_at=(
                str(row["last_accessed_at"])
                if row["last_accessed_at"] is not None
                else None
            ),
            status=MemoryStatus(str(row["status"])),
        )
