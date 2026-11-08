"""Tests for long-term memory integration with Ollie's chat pipeline."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import main
from memory.memory_models import MemoryCategory, MemoryInput
from memory.memory_service import MemoryService
from memory.memory_store import MemoryStore
from rag.qa_service import answer_question


class MemoryChatTests(unittest.TestCase):
    """Verify memory context and background maintenance do not disrupt chat."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        root = Path(self.temporary_directory.name)
        self.database_path_patcher = patch.object(main, "DB_PATH", root / "ollie.db")
        self.chroma_path_patcher = patch.object(main, "CHROMA_DB_PATH", root / "chroma")
        self.tool_selector_patcher = patch(
            "agent.agent_service.ollama.chat",
            return_value={
                "message": {
                    "content": (
                        '{"action":"final","tool_name":null,"arguments":{}}'
                    )
                }
            },
        )
        self.database_path_patcher.start()
        self.chroma_path_patcher.start()
        self.tool_selector_patcher.start()
        self.addCleanup(self.tool_selector_patcher.stop)
        self.addCleanup(self.chroma_path_patcher.stop)
        self.addCleanup(self.database_path_patcher.stop)
        main.initialize_database()
        self.client = TestClient(main.app)
        self.addCleanup(self.client.close)

    @staticmethod
    def answer_response() -> dict[str, object]:
        return {
            "response": "A test response.",
            "used_documents": False,
            "sources": [],
        }

    def memory_service(self) -> MemoryService:
        return MemoryService(MemoryStore(main.DB_PATH))

    def post_chat(self, message: str, **fields: object):
        return self.client.post(
            "/chat",
            json={"message": message, "use_documents": False, **fields},
        )

    def test_only_relevant_active_memories_are_sent_to_chat(self) -> None:
        service = self.memory_service()
        relevant = service.add_memory(
            "The user prefers concise Python answers.",
            category=MemoryCategory.PREFERENCES,
            importance=0.8,
        )
        service.add_memory(
            "The user enjoys hiking on weekends.",
            category=MemoryCategory.PERSONAL,
            importance=0.9,
        )
        archived = service.add_memory(
            "The user prefers detailed Python answers.",
            category=MemoryCategory.PREFERENCES,
            importance=1.0,
        )
        service.archive_memory(archived.id)

        with patch("main.answer_question", return_value=self.answer_response()) as answer:
            with patch("main.extract_memories", return_value=[]):
                response = self.post_chat("What Python answer style should you use?")

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(answer.call_args.kwargs["memory_context"], [relevant.content])

    def test_new_user_facts_are_extracted_and_saved_after_answer(self) -> None:
        candidate = MemoryInput(
            "The user is pursuing a B.Tech in AI and ML.",
            category=MemoryCategory.EDUCATION,
            importance=0.85,
        )
        message = "I'm pursuing a B.Tech in AI and ML."
        with patch("main.answer_question", return_value=self.answer_response()):
            with patch("main.extract_memories", return_value=[candidate]) as extractor:
                response = self.post_chat(message, conversation_id="education-chat")

        self.assertEqual(response.status_code, 200, response.text)
        extractor.assert_called_once_with(
            message,
            source_conversation_id="education-chat",
        )
        saved = self.memory_service().get_memories(category="education")
        self.assertEqual([memory.content for memory in saved], [candidate.content])
        self.assertEqual(saved[0].source_conversation_id, "education-chat")

    def test_duplicate_extraction_does_not_create_another_memory(self) -> None:
        service = self.memory_service()
        existing = service.add_memory(
            "The user prefers Python for backend development.",
            category=MemoryCategory.PREFERENCES,
        )
        duplicate = MemoryInput(
            "The user prefers Python for backend development.",
            category=MemoryCategory.PREFERENCES,
        )

        with patch("main.answer_question", return_value=self.answer_response()):
            with patch("main.extract_memories", return_value=[duplicate]):
                response = self.post_chat("I prefer Python for backend development.")

        self.assertEqual(response.status_code, 200, response.text)
        memories = service.get_memories(category="preferences")
        self.assertEqual([memory.id for memory in memories], [existing.id])

    def test_extracted_correction_updates_existing_memory(self) -> None:
        service = self.memory_service()
        existing = service.add_memory(
            "The user prefers short answers.",
            category=MemoryCategory.PREFERENCES,
            importance=0.5,
        )
        candidate = MemoryInput(
            "The user now prefers detailed explanations.",
            category=MemoryCategory.PREFERENCES,
            importance=0.9,
        )
        decision = {
            "message": {
                "content": json.dumps(
                    {
                        "action": "update",
                        "target_id": existing.id,
                        "confidence": 0.99,
                        "explanation": "The user explicitly changed their preference.",
                    }
                )
            }
        }

        with patch("main.answer_question", return_value=self.answer_response()):
            with patch("main.extract_memories", return_value=[candidate]):
                with patch(
                    "memory.memory_updater.ollama.chat",
                    return_value=decision,
                ) as decision_chat:
                    response = self.post_chat(
                        "Actually, I now prefer detailed explanations."
                    )

        self.assertEqual(response.status_code, 200, response.text)
        decision_input = json.loads(
            decision_chat.call_args.kwargs["messages"][1]["content"]
        )
        self.assertTrue(decision_input["explicit_user_correction"])
        memories = service.get_memories(category="preferences")
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0].id, existing.id)
        self.assertEqual(memories[0].content, candidate.content)

    def test_chat_succeeds_when_memory_retrieval_fails(self) -> None:
        with patch("main.answer_question", return_value=self.answer_response()) as answer:
            with patch.object(
                MemoryService,
                "retrieve_memories",
                side_effect=RuntimeError("test retrieval failure"),
            ):
                with patch("main.extract_memories", return_value=[]):
                    response = self.post_chat("Tell me a general fact.")

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(answer.call_args.kwargs["memory_context"], [])

    def test_chat_succeeds_and_history_is_saved_when_extraction_fails(self) -> None:
        with patch("main.answer_question", return_value=self.answer_response()):
            with patch(
                "main.extract_memories",
                side_effect=RuntimeError("test extraction failure"),
            ) as extractor:
                response = self.post_chat(
                    "Remember that I am studying AI.",
                    conversation_id="extraction-failure",
                )

        self.assertEqual(response.status_code, 200, response.text)
        extractor.assert_called_once_with(
            "Remember that I am studying AI.",
            source_conversation_id="extraction-failure",
        )
        history = self.client.get(
            "/history",
            params={"conversation_id": "extraction-failure"},
        ).json()["messages"]
        self.assertEqual(
            [message["role"] for message in history],
            ["user", "assistant"],
        )
        self.assertEqual(
            [message["content"] for message in history],
            ["Remember that I am studying AI.", "A test response."],
        )

    def test_personal_memories_are_separate_from_document_context(self) -> None:
        document_result = {
            "id": "chunk-1",
            "text": "The uploaded guide says the library opens at nine.",
            "document_id": "document-1",
            "metadata": {"filename": "guide.pdf", "page_number": 1},
            "distance": 0.1,
            "similarity": 0.9,
        }
        with patch("rag.qa_service.search_documents", return_value=[document_result]):
            with patch(
                "rag.qa_service.ollama.chat",
                return_value={"message": {"content": "It opens at nine."}},
            ) as chat:
                result = answer_question(
                    "When does the library open?",
                    [],
                    main.SYSTEM_PROMPT,
                    use_documents=True,
                    persist_directory=main.CHROMA_DB_PATH,
                    memory_context=["The user prefers concise answers."],
                )

        self.assertTrue(result["used_documents"])
        system_messages = [
            message["content"]
            for message in chat.call_args.kwargs["messages"]
            if message["role"] == "system"
        ]
        document_context = next(
            message for message in system_messages if "<document_context>" in message
        )
        personal_context = next(
            message
            for message in system_messages
            if "<personal_memory_context>" in message
        )
        self.assertIn("opens at nine", document_context)
        self.assertIn("prefers concise answers", personal_context)
        self.assertIn("document evidence takes precedence", personal_context)


if __name__ == "__main__":
    unittest.main()
