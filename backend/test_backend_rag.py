"""Integration tests for PDF ingestion, chat, and document-grounded answers."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fitz
from fastapi.testclient import TestClient

import main
from rag.qa_service import NO_DOCUMENT_ANSWER, answer_question
from rag.vector_store import VectorStore


def create_pdf(text: str) -> bytes:
    """Create a small in-memory PDF for upload tests."""
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    content = document.tobytes()
    document.close()
    return content


class RagBackendTests(unittest.TestCase):
    """Exercise HTTP and service integration using isolated SQLite and Chroma stores."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        root = Path(self.temporary_directory.name)
        self.database_path = root / "ollie_test.db"
        self.chroma_path = root / "chroma_db"
        self.upload_path = root / "uploads"
        self.db_path_patcher = patch.object(main, "DB_PATH", self.database_path)
        self.chroma_path_patcher = patch.object(main, "CHROMA_DB_PATH", self.chroma_path)
        self.upload_path_patcher = patch.object(
            main,
            "DOCUMENT_UPLOAD_DIRECTORY",
            self.upload_path,
        )
        self.memory_extractor_patcher = patch(
            "main.extract_memories",
            return_value=[],
        )
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
        self.db_path_patcher.start()
        self.chroma_path_patcher.start()
        self.upload_path_patcher.start()
        self.memory_extractor_patcher.start()
        self.tool_selector_patcher.start()
        self.addCleanup(self.tool_selector_patcher.stop)
        self.addCleanup(self.memory_extractor_patcher.stop)
        self.addCleanup(self.upload_path_patcher.stop)
        self.addCleanup(self.chroma_path_patcher.stop)
        self.addCleanup(self.db_path_patcher.stop)
        main.initialize_database()
        self.client = TestClient(main.app)
        self.addCleanup(self.client.close)

    @staticmethod
    def fake_document_embeddings(texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]

    @staticmethod
    def fake_chat_response(content: str = "The answer is in the document.") -> dict:
        return {"message": {"content": content}}

    def test_pdf_upload_ingests_and_stores_chunks(self) -> None:
        pdf = create_pdf(
            "The local library opens at nine in the morning and closes at six in the evening."
        )
        with patch(
            "rag.vector_store.embed_texts",
            side_effect=self.fake_document_embeddings,
        ):
            response = self.client.post(
                "/documents?filename=library.pdf",
                content=pdf,
                headers={"content-type": "application/pdf"},
            )

        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["filename"], "library.pdf")
        self.assertEqual(result["page_count"], 1)
        self.assertGreater(result["chunk_count"], 0)
        self.assertEqual(len(result["chunk_ids"]), result["chunk_count"])
        with VectorStore(self.chroma_path) as store:
            self.assertEqual(store.get_chunk_count(), result["chunk_count"])
            records = store.collection.get(
                ids=result["chunk_ids"],
                include=["documents", "metadatas"],
            )
        self.assertIn("opens at nine", records["documents"][0])
        self.assertEqual(records["metadatas"][0]["document_id"], result["document_id"])
        self.assertEqual(records["metadatas"][0]["filename"], "library.pdf")

    def test_upload_rejects_non_pdf_content_and_invalid_pdf(self) -> None:
        wrong_type = self.client.post(
            "/documents?filename=notes.pdf",
            content=b"not a pdf",
            headers={"content-type": "text/plain"},
        )
        self.assertEqual(wrong_type.status_code, 415)

        invalid_pdf = self.client.post(
            "/documents?filename=notes.pdf",
            content=b"%PDF- malformed document data",
            headers={"content-type": "application/pdf"},
        )
        self.assertEqual(invalid_pdf.status_code, 400)

    def test_answerable_question_uses_retrieved_context(self) -> None:
        matches = [
            {
                "id": "chunk-1",
                "text": "The library opens at nine in the morning.",
                "document_id": "doc-1",
                "metadata": {
                    "filename": "library.pdf",
                    "page_number": 1,
                },
                "distance": 0.1,
                "similarity": 0.9,
            }
        ]
        with patch("rag.qa_service.search_documents", return_value=matches):
            with patch(
                "rag.qa_service.ollama.chat",
                return_value=self.fake_chat_response("It opens at nine in the morning."),
            ) as chat:
                answer = answer_question(
                    "When does the library open?",
                    [],
                    main.SYSTEM_PROMPT,
                    use_documents=None,
                    persist_directory=self.chroma_path,
                )

        self.assertEqual(answer["response"], "It opens at nine in the morning.")
        self.assertTrue(answer["used_documents"])
        self.assertEqual(answer["sources"][0]["filename"], "library.pdf")
        messages = chat.call_args.kwargs["messages"]
        context_message = next(
            message for message in messages if "<document_context>" in message["content"]
        )
        self.assertIn("opens at nine", context_message["content"])

    def test_unanswerable_document_question_returns_clear_no_context_message(self) -> None:
        with patch("rag.qa_service.search_documents", return_value=[]):
            with patch("rag.qa_service.ollama.chat") as chat:
                answer = answer_question(
                    "What is the library's street address?",
                    [],
                    main.SYSTEM_PROMPT,
                    use_documents=True,
                    persist_directory=self.chroma_path,
                )

        self.assertEqual(answer["response"], NO_DOCUMENT_ANSWER)
        self.assertFalse(answer["used_documents"])
        chat.assert_not_called()

    def test_general_chat_skips_retrieval_without_documents(self) -> None:
        with patch("rag.qa_service.search_documents") as search:
            with patch(
                "rag.qa_service.ollama.chat",
                return_value=self.fake_chat_response("Hello!"),
            ) as chat:
                answer = answer_question(
                    "Hello Ollie",
                    [{"role": "user", "content": "Hi"}],
                    main.SYSTEM_PROMPT,
                    use_documents=False,
                    persist_directory=self.chroma_path,
                )

        search.assert_not_called()
        self.assertEqual(answer["response"], "Hello!")
        self.assertFalse(answer["used_documents"])
        messages = chat.call_args.kwargs["messages"]
        self.assertEqual(messages[-2], {"role": "user", "content": "Hi"})

    def test_default_chat_falls_back_when_collection_is_missing(self) -> None:
        with patch(
            "rag.qa_service.ollama.chat",
            return_value=self.fake_chat_response("A general response."),
        ):
            answer = answer_question(
                "Tell me a fun fact.",
                [],
                main.SYSTEM_PROMPT,
                persist_directory=self.chroma_path,
            )
        self.assertEqual(answer["response"], "A general response.")
        self.assertFalse(answer["used_documents"])

    def test_chat_preserves_conversation_history_and_conversation_id(self) -> None:
        answers = [
            {
                "response": "First response.",
                "used_documents": False,
                "sources": [],
            },
            {
                "response": "Second response.",
                "used_documents": False,
                "sources": [],
            },
        ]
        with patch("main.answer_question", side_effect=answers) as answer_mock:
            first = self.client.post(
                "/chat",
                json={"message": "First question", "conversation_id": "conversation-a"},
            )
            second = self.client.post(
                "/chat",
                json={"message": "Follow-up", "conversation_id": "conversation-a"},
            )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.json()["conversation_id"], "conversation-a")
        follow_up_history = answer_mock.call_args_list[1].kwargs["history"]
        self.assertEqual(
            follow_up_history,
            [
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "First response."},
            ],
        )
        history = self.client.get(
            "/history",
            params={"conversation_id": "conversation-a"},
        ).json()
        self.assertEqual(
            [message["content"] for message in history["messages"]],
            ["First question", "First response.", "Follow-up", "Second response."],
        )
        other_history = self.client.get(
            "/history",
            params={"conversation_id": "conversation-b"},
        ).json()
        self.assertEqual(other_history["messages"], [])

    def test_end_to_end_pdf_upload_then_document_question(self) -> None:
        pdf = create_pdf(
            "Ollie's office is on Cedar Street. It opens at nine in the morning."
        )
        with patch(
            "rag.vector_store.embed_texts",
            side_effect=self.fake_document_embeddings,
        ):
            upload_response = self.client.post(
                "/documents?filename=office-guide.pdf",
                content=pdf,
                headers={"content-type": "application/pdf"},
            )
        self.assertEqual(upload_response.status_code, 200, upload_response.text)

        with patch("rag.retrieval.embed_text", return_value=[1.0, 0.0, 0.0]):
            with patch(
                "rag.qa_service.ollama.chat",
                return_value=self.fake_chat_response(
                    "Ollie's office is on Cedar Street and opens at nine."
                ),
            ) as chat:
                chat_response = self.client.post(
                    "/chat",
                    json={
                        "message": "Where is Ollie's office and when does it open?",
                        "conversation_id": "rag-conversation",
                        "use_documents": True,
                    },
                )

        self.assertEqual(chat_response.status_code, 200, chat_response.text)
        body = chat_response.json()
        self.assertTrue(body["used_documents"])
        self.assertEqual(body["sources"][0]["filename"], "office-guide.pdf")
        self.assertIn("Cedar Street", body["response"])
        self.assertIn(
            "Cedar Street",
            chat.call_args.kwargs["messages"][1]["content"],
        )
        history = self.client.get(
            "/history",
            params={"conversation_id": "rag-conversation"},
        ).json()
        self.assertEqual(len(history["messages"]), 2)
        self.assertEqual(history["messages"][0]["role"], "user")
        self.assertEqual(history["messages"][1]["role"], "assistant")


if __name__ == "__main__":
    unittest.main()
