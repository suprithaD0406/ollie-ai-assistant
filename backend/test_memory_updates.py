"""Tests for memory reconciliation and safe update decisions."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import ollama

from memory.memory_models import MemoryCategory, MemoryInput, MemoryStatus
from memory.memory_service import MemoryNotFoundError, MemoryService
from memory.memory_store import MemoryStore
from memory.memory_updater import (
    MemoryUpdateAction,
    MemoryUpdateError,
)


def decision_response(
    action: str,
    target_id: str | None = None,
    confidence: float = 0.99,
    explanation: str = "Test decision.",
) -> dict[str, object]:
    """Build the JSON response shape expected from Ollama."""
    return {
        "message": {
            "content": json.dumps(
                {
                    "action": action,
                    "target_id": target_id,
                    "confidence": confidence,
                    "explanation": explanation,
                }
            )
        }
    }


class MemoryUpdateTests(unittest.TestCase):
    """Verify additions, replacements, conflicts, and preservation rules."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        database_path = Path(self.temporary_directory.name) / "ollie.db"
        self.store = MemoryStore(database_path)
        self.service = MemoryService(self.store)

    def decide(self, action: str, target_id: str | None = None, **kwargs: object):
        return patch(
            "memory.memory_updater.ollama.chat",
            return_value=decision_response(action, target_id, **kwargs),
        )

    def test_new_memory_is_created_without_model_call_when_category_is_empty(self) -> None:
        incoming = MemoryInput(
            "The user is pursuing a B.Tech in AI and ML.",
            category=MemoryCategory.EDUCATION,
            importance=0.8,
            source_conversation_id="conversation-new",
        )
        with patch("memory.memory_updater.ollama.chat") as chat:
            result = self.service.apply_memory(incoming)

        chat.assert_not_called()
        self.assertEqual(result.action, MemoryUpdateAction.CREATED)
        self.assertEqual(result.memory.content, incoming.content)
        self.assertEqual(result.memory.status, MemoryStatus.ACTIVE)

    def test_exact_duplicate_is_not_stored_twice(self) -> None:
        original = self.service.add_memory(
            "I prefer short answers.",
            category=MemoryCategory.PREFERENCES,
            importance=0.7,
        )
        with patch("memory.memory_updater.ollama.chat") as chat:
            result = self.service.apply_memory(
                MemoryInput(
                    "  i   prefer SHORT answers. ",
                    category=MemoryCategory.OTHER,
                    importance=0.9,
                )
            )

        chat.assert_not_called()
        self.assertEqual(result.action, MemoryUpdateAction.DUPLICATE)
        self.assertEqual(result.memory.id, original.id)
        self.assertEqual(len(self.service.get_memories(include_archived=True)), 1)

    def test_clear_semester_replacement_preserves_identity_and_created_at(self) -> None:
        existing = self.service.add_memory(
            "The user is in 4th semester.",
            category=MemoryCategory.EDUCATION,
            importance=0.5,
            source_conversation_id="conversation-old",
        )
        with self.decide("update", existing.id):
            result = self.service.apply_memory(
                MemoryInput(
                    "The user has entered 5th semester.",
                    category=MemoryCategory.EDUCATION,
                    importance=0.9,
                    source_conversation_id="conversation-new",
                )
            )

        self.assertEqual(result.action, MemoryUpdateAction.UPDATED)
        self.assertEqual(result.memory.id, existing.id)
        self.assertEqual(result.memory.created_at, existing.created_at)
        self.assertGreaterEqual(result.memory.updated_at, existing.updated_at)
        self.assertEqual(result.memory.content, "The user has entered 5th semester.")
        self.assertEqual(result.memory.source_conversation_id, "conversation-new")
        self.assertEqual(len(self.service.get_memories(category="education")), 1)

    def test_changed_preference_replaces_previous_preference(self) -> None:
        existing = self.service.add_memory(
            "The user prefers short answers.",
            category=MemoryCategory.PREFERENCES,
        )
        with self.decide("update", existing.id):
            result = self.service.apply_memory(
                MemoryInput(
                    "The user now prefers detailed explanations.",
                    category=MemoryCategory.PREFERENCES,
                    importance=0.8,
                )
            )

        self.assertEqual(result.action, MemoryUpdateAction.UPDATED)
        self.assertEqual(result.memory.id, existing.id)
        self.assertEqual(result.memory.content, "The user now prefers detailed explanations.")
        self.assertEqual(len(self.service.get_memories(category="preferences")), 1)

    def test_additional_fact_is_preserved_as_a_separate_memory(self) -> None:
        python = self.service.add_memory(
            "The user knows Python.",
            category=MemoryCategory.CAREER,
            importance=0.8,
        )
        with self.decide("add"):
            result = self.service.apply_memory(
                MemoryInput(
                    "The user also knows Java.",
                    category=MemoryCategory.CAREER,
                    importance=0.7,
                )
            )

        self.assertEqual(result.action, MemoryUpdateAction.CREATED)
        self.assertNotEqual(result.memory.id, python.id)
        self.assertEqual(
            {memory.content for memory in self.service.get_memories(category="career")},
            {"The user knows Python.", "The user also knows Java."},
        )

    def test_ambiguous_conflict_does_not_overwrite_or_create(self) -> None:
        existing = self.service.add_memory(
            "The user plans to work in Boston.",
            category=MemoryCategory.PERSONAL,
            importance=0.7,
        )
        with self.decide("ambiguous", confidence=0.96, explanation="It may be a conflict."):
            result = self.service.apply_memory(
                MemoryInput(
                    "The user plans to work in Seattle.",
                    category=MemoryCategory.PERSONAL,
                    importance=0.8,
                )
            )

        self.assertEqual(result.action, MemoryUpdateAction.NEEDS_REVIEW)
        self.assertIsNone(result.memory)
        current = self.service.get_memory(existing.id)
        self.assertEqual(current.content, existing.content)
        self.assertEqual(len(self.service.get_memories(category="personal")), 1)

    def test_explicit_correction_is_passed_to_decision_and_takes_priority(self) -> None:
        existing = self.service.add_memory(
            "The user works as a designer.",
            category=MemoryCategory.CAREER,
        )
        with self.decide("update", existing.id) as chat:
            result = self.service.apply_memory(
                MemoryInput(
                    "Correction: I work as a software developer.",
                    category=MemoryCategory.CAREER,
                    source_conversation_id="correction-conversation",
                ),
                explicit_correction=True,
            )
        decision_input = json.loads(chat.call_args.kwargs["messages"][1]["content"])
        self.assertTrue(decision_input["explicit_user_correction"])
        self.assertEqual(result.action, MemoryUpdateAction.UPDATED)
        self.assertEqual(result.memory.id, existing.id)
        self.assertEqual(
            result.memory.source_conversation_id,
            "correction-conversation",
        )

    def test_unrelated_records_remain_unchanged(self) -> None:
        related = self.service.add_memory(
            "The user is in 4th semester of college.",
            category=MemoryCategory.EDUCATION,
            importance=0.6,
        )
        unrelated = self.service.add_memory(
            "The user has a degree in biology.",
            category=MemoryCategory.EDUCATION,
            importance=0.9,
        )
        with self.decide("update", related.id):
            result = self.service.apply_memory(
                MemoryInput(
                    "The user has entered 5th semester of college.",
                    category=MemoryCategory.EDUCATION,
                    importance=0.8,
                )
            )

        self.assertEqual(result.action, MemoryUpdateAction.UPDATED)
        unchanged = self.service.get_memory(unrelated.id)
        self.assertEqual(unchanged.content, unrelated.content)
        self.assertEqual(unchanged.created_at, unrelated.created_at)
        self.assertEqual(unchanged.updated_at, unrelated.updated_at)

    def test_archived_memory_is_not_a_candidate_unless_explicitly_restored(self) -> None:
        archived = self.service.add_memory(
            "The user is in 4th semester.",
            category=MemoryCategory.EDUCATION,
        )
        self.service.archive_memory(archived.id)

        with patch("memory.memory_updater.ollama.chat") as chat:
            result = self.service.apply_memory(
                MemoryInput(
                    "The user has entered 5th semester.",
                    category=MemoryCategory.EDUCATION,
                )
            )
        chat.assert_not_called()
        self.assertEqual(result.action, MemoryUpdateAction.CREATED)
        self.assertEqual(self.service.get_memory(archived.id).status, MemoryStatus.ARCHIVED)

        with self.assertRaises(MemoryNotFoundError):
            self.service.update_memory(archived.id, content="Archived stays archived.")
        restored = self.service.update_memory(
            archived.id,
            content="The user is in 4th semester.",
            restore_archived=True,
        )
        self.assertEqual(restored.id, archived.id)
        self.assertEqual(restored.created_at, archived.created_at)
        self.assertEqual(restored.status, MemoryStatus.ACTIVE)

    def test_missing_memory_id_is_reported(self) -> None:
        with self.assertRaises(MemoryNotFoundError):
            self.service.update_memory("missing-id", content="Updated fact.")

    def test_persistence_after_reopening_database(self) -> None:
        existing = self.service.add_memory(
            "The user is in 4th semester.",
            category=MemoryCategory.EDUCATION,
            source_conversation_id="old-conversation",
        )
        with self.decide("update", existing.id):
            updated = self.service.apply_memory(
                MemoryInput(
                    "The user has entered 5th semester.",
                    category=MemoryCategory.EDUCATION,
                    source_conversation_id="new-conversation",
                )
            )

        reopened = MemoryService(MemoryStore(self.store.database_path))
        persisted = reopened.get_memory(existing.id)
        self.assertEqual(updated.memory.id, existing.id)
        self.assertEqual(persisted.content, "The user has entered 5th semester.")
        self.assertEqual(persisted.created_at, existing.created_at)
        self.assertEqual(persisted.source_conversation_id, "new-conversation")

    def test_invalid_or_untrusted_model_decisions_never_update(self) -> None:
        existing = self.service.add_memory(
            "The user is learning Python.",
            category=MemoryCategory.EDUCATION,
        )
        with patch(
            "memory.memory_updater.ollama.chat",
            return_value=decision_response("update", "not-a-candidate"),
        ):
            with self.assertRaises(MemoryUpdateError):
                self.service.apply_memory(
                    MemoryInput(
                        "The user is learning Java.",
                        category=MemoryCategory.EDUCATION,
                    )
                )
        self.assertEqual(self.service.get_memory(existing.id).content, existing.content)

    def test_low_confidence_update_is_sent_for_review(self) -> None:
        existing = self.service.add_memory(
            "The user is in 4th semester.",
            category=MemoryCategory.EDUCATION,
        )
        with self.decide("update", existing.id, confidence=0.5):
            result = self.service.apply_memory(
                MemoryInput(
                    "The user is in 5th semester.",
                    category=MemoryCategory.EDUCATION,
                )
            )
        self.assertEqual(result.action, MemoryUpdateAction.NEEDS_REVIEW)
        self.assertEqual(self.service.get_memory(existing.id).content, existing.content)

    def test_ollama_connection_error_does_not_change_storage(self) -> None:
        existing = self.service.add_memory(
            "The user is in 4th semester.",
            category=MemoryCategory.EDUCATION,
        )
        with patch(
            "memory.memory_updater.ollama.chat",
            side_effect=ollama.RequestError("Ollama unavailable"),
        ):
            with self.assertRaises(MemoryUpdateError):
                self.service.apply_memory(
                    MemoryInput(
                        "The user is in 5th semester.",
                        category=MemoryCategory.EDUCATION,
                    )
                )
        self.assertEqual(self.service.get_memory(existing.id).content, existing.content)


if __name__ == "__main__":
    unittest.main()
