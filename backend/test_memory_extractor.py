"""Tests for structured long-term memory extraction."""

import json
import unittest
from unittest.mock import patch

import ollama

from memory.memory_extractor import MemoryExtractionError, extract_memories
from memory.memory_models import MemoryCategory, MemoryInput


def ollama_response(payload: object) -> dict[str, object]:
    """Build the shape returned by the Ollama chat client."""
    return {"message": {"content": json.dumps(payload)}}


class MemoryExtractorTests(unittest.TestCase):
    """Test model prompting, output validation, filtering, and local integration."""

    def test_extracts_education_fact(self) -> None:
        with patch(
            "memory.memory_extractor.ollama.chat",
            return_value=ollama_response(
                {
                    "memories": [
                        {
                            "content": "The user is pursuing a B.Tech in AI and ML.",
                            "category": "education",
                            "importance": 0.85,
                        }
                    ]
                }
            ),
        ) as chat:
            memories = extract_memories(
                "I'm pursuing B.Tech in AI and ML.",
                source_conversation_id="conversation-1",
            )

        self.assertEqual(len(memories), 1)
        self.assertIsInstance(memories[0], MemoryInput)
        self.assertEqual(memories[0].category, MemoryCategory.EDUCATION)
        self.assertEqual(memories[0].source_conversation_id, "conversation-1")
        self.assertEqual(chat.call_args.kwargs["model"], "qwen2.5:3b")
        self.assertIn("format", chat.call_args.kwargs)

    def test_extracts_multiple_distinct_facts(self) -> None:
        with patch(
            "memory.memory_extractor.ollama.chat",
            return_value=ollama_response(
                {
                    "memories": [
                        {
                            "content": "The user is studying computer science.",
                            "category": "education",
                            "importance": 0.8,
                        },
                        {
                            "content": "The user prefers Python for backend development.",
                            "category": "preferences",
                            "importance": 0.75,
                        },
                    ]
                }
            ),
        ):
            memories = extract_memories(
                "I'm studying computer science and prefer Python for backend development."
            )

        self.assertEqual(len(memories), 2)
        self.assertEqual(
            {memory.category for memory in memories},
            {MemoryCategory.EDUCATION, MemoryCategory.PREFERENCES},
        )

    def test_irrelevant_message_returns_no_memories(self) -> None:
        with patch(
            "memory.memory_extractor.ollama.chat",
            return_value=ollama_response({"memories": []}),
        ) as chat:
            memories = extract_memories("What is the difference between TCP and UDP?")

        self.assertEqual(memories, [])
        chat.assert_called_once()

    def test_sensitive_data_is_filtered_without_calling_ollama(self) -> None:
        with patch("memory.memory_extractor.ollama.chat") as chat:
            memories = extract_memories("My password is example-password-123.")

        self.assertEqual(memories, [])
        chat.assert_not_called()

    def test_sensitive_output_is_filtered_by_default(self) -> None:
        with patch(
            "memory.memory_extractor.ollama.chat",
            return_value=ollama_response(
                {
                    "memories": [
                        {
                            "content": "The user's email is ollie@example.com.",
                            "category": "personal",
                            "importance": 0.9,
                        }
                    ]
                }
            ),
        ):
            self.assertEqual(extract_memories("Please remember how to contact me."), [])

    def test_malformed_llm_json_is_reported(self) -> None:
        with patch(
            "memory.memory_extractor.ollama.chat",
            return_value={"message": {"content": "not valid json"}},
        ):
            with self.assertRaisesRegex(MemoryExtractionError, "malformed"):
                extract_memories("I prefer Python.")

    def test_invalid_category_is_reported(self) -> None:
        with patch(
            "memory.memory_extractor.ollama.chat",
            return_value=ollama_response(
                {
                    "memories": [
                        {
                            "content": "The user prefers Python.",
                            "category": "hobbies",
                            "importance": 0.7,
                        }
                    ]
                }
            ),
        ):
            with self.assertRaisesRegex(MemoryExtractionError, "invalid memory"):
                extract_memories("I prefer Python.")

    def test_invalid_importance_and_output_shape_are_reported(self) -> None:
        invalid_importance = ollama_response(
            {
                "memories": [
                    {
                        "content": "The user prefers Python.",
                        "category": "preferences",
                        "importance": 2,
                    }
                ]
            }
        )
        with patch("memory.memory_extractor.ollama.chat", return_value=invalid_importance):
            with self.assertRaisesRegex(MemoryExtractionError, "invalid memory"):
                extract_memories("I prefer Python.")

        with patch(
            "memory.memory_extractor.ollama.chat",
            return_value=ollama_response({"facts": []}),
        ):
            with self.assertRaisesRegex(MemoryExtractionError, "memories"):
                extract_memories("I prefer Python.")

    def test_duplicate_extractions_are_deduplicated(self) -> None:
        with patch(
            "memory.memory_extractor.ollama.chat",
            return_value=ollama_response(
                {
                    "memories": [
                        {
                            "content": "The user prefers Python.",
                            "category": "preferences",
                            "importance": 0.7,
                        },
                        {
                            "content": "  the user   PREFERS Python. ",
                            "category": "preferences",
                            "importance": 0.6,
                        },
                    ]
                }
            ),
        ):
            memories = extract_memories("I prefer Python.")

        self.assertEqual(len(memories), 1)

    def test_ollama_connection_error_is_reported(self) -> None:
        with patch(
            "memory.memory_extractor.ollama.chat",
            side_effect=ollama.RequestError("Ollama is unavailable"),
        ):
            with self.assertRaisesRegex(MemoryExtractionError, "Could not extract"):
                extract_memories("I prefer Python.")

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            extract_memories(" ")
        with self.assertRaises(TypeError):
            extract_memories(None)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            extract_memories("Remember this", allow_sensitive="yes")  # type: ignore[arg-type]

    def test_real_local_ollama_extraction(self) -> None:
        memories = extract_memories(
            "I am pursuing a B.Tech degree in artificial intelligence and machine learning."
        )

        self.assertTrue(memories, "Expected Ollama to extract the clearly stated education fact.")
        self.assertTrue(
            any(memory.category == MemoryCategory.EDUCATION for memory in memories),
            f"Expected an education memory, got {memories!r}.",
        )
        self.assertTrue(all(isinstance(memory, MemoryInput) for memory in memories))
        self.assertTrue(all(0.0 <= memory.importance <= 1.0 for memory in memories))


if __name__ == "__main__":
    unittest.main()
