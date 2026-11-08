"""Tests for registered local tools and bounded agent orchestration."""

import json
import copy
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import ollama

from agent.agent_service import AgentService
from agent.tool_executor import ToolExecutor
from agent.tool_registry import ToolRegistry, ToolRegistryError, ToolSpec
from agent.tool_result_handler import ToolResultHandler
from agent.tools import build_default_tool_registry
from memory.memory_models import MemoryCategory
from memory.memory_service import MemoryService
from memory.memory_store import MemoryStore


def selection(action: str, tool_name: str | None = None, arguments=None) -> dict:
    """Create a native Ollama function-call response."""
    if action == "final":
        return {"message": {"content": "No tool is needed.", "tool_calls": []}}
    return {
        "message": {
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": tool_name,
                        "arguments": {} if arguments is None else arguments,
                    }
                }
            ],
        }
    }


class ToolRegistryTests(unittest.TestCase):
    def test_registration_discovery_and_duplicate_rejection(self) -> None:
        registry = ToolRegistry()
        tool = ToolSpec(
            "sample",
            "A sample tool.",
            {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            lambda _arguments: {"ok": True},
        )
        registry.register(tool)

        self.assertIs(registry.get("sample"), tool)
        self.assertEqual(registry.descriptions()[0]["name"], "sample")
        with self.assertRaises(ToolRegistryError):
            registry.register(tool)

    def test_initial_tool_discovery_contains_only_four_registered_tools(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            registry = build_default_tool_registry(
                root / "ollie.db",
                root / "chroma",
                user_message="What time is it?",
                conversation_id="tool-test",
            )

        self.assertEqual(
            {tool.name for tool in registry.list_tools()},
            {
                "get_current_datetime",
                "search_uploaded_documents",
                "retrieve_user_memories",
                "save_user_memory",
            },
        )


class ToolExecutorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = ToolRegistry()
        self.registry.register(
            ToolSpec(
                "echo",
                "Return the supplied value.",
                {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string", "maxLength": 20},
                    },
                    "required": ["value"],
                    "additionalProperties": False,
                },
                lambda arguments: {"value": arguments["value"]},
            )
        )

    def test_valid_registered_execution(self) -> None:
        result = ToolExecutor(self.registry).execute(
            "echo",
            {"value": "hello"},
            user_message="Say hello.",
        )
        self.assertTrue(result.success)
        self.assertEqual(result.data, {"value": "hello"})

    def test_invalid_arguments_and_unknown_tools_are_rejected(self) -> None:
        executor = ToolExecutor(self.registry)
        invalid = executor.execute(
            "echo",
            {"value": "hello", "extra": True},
            user_message="Say hello.",
        )
        unknown = executor.execute(
            "shell",
            {"command": "not allowed"},
            user_message="Run a command.",
        )
        self.assertFalse(invalid.success)
        self.assertEqual(invalid.error, "Tool arguments contain unsupported fields.")
        self.assertFalse(unknown.success)
        self.assertEqual(unknown.error, "Unknown tool.")

    def test_tool_errors_are_sanitized(self) -> None:
        self.registry.register(
            ToolSpec(
                "failure",
                "Raise an exception.",
                {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
                lambda _arguments: (_ for _ in ()).throw(
                    RuntimeError("sensitive exception details")
                ),
            )
        )
        result = ToolExecutor(self.registry).execute(
            "failure",
            {},
            user_message="Try the tool.",
        )
        self.assertFalse(result.success)
        self.assertEqual(result.error, "Tool execution failed.")
        self.assertNotIn(
            "sensitive",
            ToolResultHandler.for_model(result),
        )

    def test_tool_timeout_is_reported(self) -> None:
        slow_registry = ToolRegistry()
        slow_registry.register(
            ToolSpec(
                "slow",
                "A slow tool.",
                {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
                lambda _arguments: time.sleep(0.05),
            )
        )
        result = ToolExecutor(slow_registry, timeout_seconds=0.005).execute(
            "slow",
            {},
            user_message="Try a slow tool.",
        )
        self.assertFalse(result.success)
        self.assertEqual(result.error, "Tool execution timed out.")
        time.sleep(0.06)

    def test_save_tool_requires_explicit_intent_and_filters_sensitive_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            registry = build_default_tool_registry(
                root / "ollie.db",
                root / "chroma",
                user_message="Tell me about passwords.",
                conversation_id="safety-test",
            )
            executor = ToolExecutor(registry)
            unauthorized = executor.execute(
                "save_user_memory",
                {
                    "content": "The user prefers clear answers.",
                    "category": "preferences",
                    "importance": 0.8,
                },
                user_message="Tell me about passwords.",
            )
            mentioned_only = executor.execute(
                "save_user_memory",
                {
                    "content": "The user prefers clear answers.",
                    "category": "preferences",
                    "importance": 0.8,
                },
                user_message="What does the phrase 'remember this' mean?",
            )
            sensitive_registry = build_default_tool_registry(
                root / "sensitive.db",
                root / "chroma",
                user_message="Remember this.",
                conversation_id="safety-test",
            )
            sensitive = ToolExecutor(sensitive_registry).execute(
                "save_user_memory",
                {
                    "content": "The user's password is hunter2.",
                    "category": "personal",
                    "importance": 0.9,
                },
                user_message="Remember this password.",
            )

        self.assertFalse(unauthorized.success)
        self.assertIn("explicitly ask", unauthorized.error)
        self.assertFalse(mentioned_only.success)
        self.assertFalse(sensitive.success)
        self.assertEqual(sensitive.error, "Tool execution failed.")


class AgentServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = ToolRegistry()
        self.registry.register(
            ToolSpec(
                "first",
                "Return the first result.",
                {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
                lambda _arguments: {"value": "one"},
            )
        )
        self.registry.register(
            ToolSpec(
                "second",
                "Return the second result.",
                {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
                lambda _arguments: {"value": "two"},
            )
        )
        self.answer = {
            "response": "Done.",
            "used_documents": False,
            "sources": [],
        }

    def call_agent(self, max_tool_calls: int = 3) -> dict[str, object]:
        service = AgentService(
            self.registry,
            answerer=lambda **_kwargs: self.answer,
            max_tool_calls=max_tool_calls,
        )
        return service.respond(
            question="Run the useful tools.",
            history=[],
            system_prompt="You are Ollie.",
            use_documents=False,
        )

    def test_normal_response_without_tools(self) -> None:
        with patch(
            "agent.agent_service.ollama.chat",
            return_value=selection("final"),
        ) as chat:
            self.call_agent()

        self.assertEqual(chat.call_count, 1)

    def test_multiple_tool_calls_are_returned_to_the_selector_and_answerer(self) -> None:
        decisions = [
            selection("tool_call", "first"),
            selection("tool_call", "second"),
            selection("final"),
        ]
        captured: dict[str, object] = {}

        def answerer(**kwargs):
            captured.update(kwargs)
            return self.answer

        service = AgentService(self.registry, answerer=answerer)
        observed_messages = []

        def select(*, messages, **_kwargs):
            observed_messages.append(copy.deepcopy(messages))
            return decisions.pop(0)

        with patch(
            "agent.agent_service.ollama.chat",
            side_effect=select,
        ) as chat:
            service.respond(
                question="Run the useful tools.",
                history=[],
                system_prompt="You are Ollie.",
                use_documents=False,
            )

        self.assertEqual(chat.call_count, 3)
        self.assertIn("one", observed_messages[1][-1]["content"])
        self.assertEqual(len(captured["tool_context"]), 2)
        self.assertIn("two", captured["tool_context"][1])

    def test_maximum_tool_calls_is_enforced(self) -> None:
        service = AgentService(
            self.registry,
            answerer=lambda **_kwargs: self.answer,
            max_tool_calls=1,
        )
        with patch(
            "agent.agent_service.ollama.chat",
            return_value=selection("tool_call", "first"),
        ) as chat:
            service.respond(
                question="Use tools.",
                history=[],
                system_prompt="You are Ollie.",
                use_documents=False,
            )

        self.assertEqual(chat.call_count, 1)

    def test_unknown_tool_result_is_safe_and_model_can_continue(self) -> None:
        decisions = [
            selection("tool_call", "unregistered"),
            selection("final"),
        ]
        captured: dict[str, object] = {}

        def answerer(**kwargs):
            captured.update(kwargs)
            return self.answer

        service = AgentService(self.registry, answerer=answerer)
        with patch("agent.agent_service.ollama.chat", side_effect=decisions):
            service.respond(
                question="Use the unknown tool.",
                history=[],
                system_prompt="You are Ollie.",
                use_documents=False,
            )
        self.assertIn("Unknown tool.", captured["tool_context"][0])

    def test_malformed_selection_falls_back_to_normal_answering(self) -> None:
        captured: dict[str, object] = {}

        def answerer(**kwargs):
            captured.update(kwargs)
            return self.answer

        service = AgentService(self.registry, answerer=answerer)
        with patch(
            "agent.agent_service.ollama.chat",
            return_value={
                "message": {
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "first", "arguments": "{}"}}
                    ],
                }
            },
        ):
            result = service.respond(
                question="Just answer normally.",
                history=[],
                system_prompt="You are Ollie.",
                use_documents=False,
            )
        self.assertEqual(result, self.answer)
        self.assertEqual(captured["tool_context"], [])

    def test_tool_selection_request_failure_falls_back_to_normal_answering(self) -> None:
        captured: dict[str, object] = {}

        def answerer(**kwargs):
            captured.update(kwargs)
            return self.answer

        service = AgentService(self.registry, answerer=answerer)
        with patch(
            "agent.agent_service.ollama.chat",
            side_effect=ollama.RequestError("server unavailable"),
        ):
            result = service.respond(
                question="Just answer normally.",
                history=[],
                system_prompt="You are Ollie.",
                use_documents=False,
            )

        self.assertEqual(result, self.answer)
        self.assertEqual(captured["tool_context"], [])

    def test_unexpected_tool_selection_errors_are_not_hidden(self) -> None:
        with patch(
            "agent.agent_service.ollama.chat",
            side_effect=ValueError("unexpected selector bug"),
        ):
            with self.assertRaisesRegex(ValueError, "unexpected selector bug"):
                self.call_agent()

    def test_datetime_tool_selection_executes_registered_tool(self) -> None:
        registry = ToolRegistry()
        registry.register(
            ToolSpec(
                "get_current_datetime",
                "Get the current local date and time.",
                {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
                lambda _arguments: {"datetime": "2026-11-08T12:00:00+05:30"},
            )
        )
        captured: dict[str, object] = {}

        def answerer(**kwargs):
            captured.update(kwargs)
            return self.answer

        service = AgentService(registry, answerer=answerer)
        with patch(
            "agent.agent_service.ollama.chat",
            side_effect=[
                selection("tool_call", "get_current_datetime", {}),
                selection("final"),
            ],
        ) as chat:
            service.respond(
                question="What time is it?",
                history=[],
                system_prompt="You are Ollie.",
                use_documents=False,
            )

        self.assertEqual(chat.call_args_list[0].kwargs["tools"][0]["function"]["name"],
                         "get_current_datetime")
        self.assertIn("2026-11-08T12:00:00+05:30", captured["tool_context"][0])

    def test_memory_retrieval_selection_passes_query_arguments(self) -> None:
        registry = ToolRegistry()
        observed_arguments: list[dict[str, object]] = []

        def retrieve(arguments):
            observed_arguments.append(arguments)
            return {"memories": ["The user prefers concise answers."]}

        registry.register(
            ToolSpec(
                "retrieve_user_memories",
                "Retrieve relevant active personal memories.",
                {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
                retrieve,
            )
        )
        captured: dict[str, object] = {}

        def answerer(**kwargs):
            captured.update(kwargs)
            return self.answer

        service = AgentService(registry, answerer=answerer)
        with patch(
            "agent.agent_service.ollama.chat",
            side_effect=[
                selection(
                    "tool_call",
                    "retrieve_user_memories",
                    {"query": "answer style"},
                ),
                selection("final"),
            ],
        ):
            service.respond(
                question="How do I prefer answers?",
                history=[],
                system_prompt="You are Ollie.",
                use_documents=False,
            )

        self.assertEqual(observed_arguments, [{"query": "answer style"}])
        self.assertIn("prefers concise answers", captured["tool_context"][0])

    def test_required_tool_arguments_are_validated_when_missing(self) -> None:
        registry = ToolRegistry()
        registry.register(
            ToolSpec(
                "search_uploaded_documents",
                "Search uploaded documents.",
                {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
                lambda _arguments: {"results": []},
            )
        )
        captured: dict[str, object] = {}

        def answerer(**kwargs):
            captured.update(kwargs)
            return self.answer

        service = AgentService(registry, answerer=answerer)
        with patch(
            "agent.agent_service.ollama.chat",
            side_effect=[
                selection("tool_call", "search_uploaded_documents", {}),
                selection("final"),
            ],
        ):
            service.respond(
                question="Search my document.",
                history=[],
                system_prompt="You are Ollie.",
                use_documents=True,
            )

        self.assertIn(
            "A required tool argument is missing.",
            captured["tool_context"][0],
        )

    def test_document_search_results_are_passed_to_qa_as_retrieved_context(self) -> None:
        registry = ToolRegistry()
        observed_arguments: list[dict[str, object]] = []

        def search(arguments):
            observed_arguments.append(arguments)
            return {
                "results": [
                    {
                        "text": "The guide says the lab opens at ten.",
                        "filename": "guide.pdf",
                        "page_number": 2,
                        "similarity": 0.9,
                    }
                ]
            }

        registry.register(
            ToolSpec(
                "search_uploaded_documents",
                "Search uploaded documents.",
                {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
                search,
            )
        )
        captured: dict[str, object] = {}

        def answerer(**kwargs):
            captured.update(kwargs)
            return self.answer

        service = AgentService(registry, answerer=answerer)
        with patch(
            "agent.agent_service.ollama.chat",
            side_effect=[
                selection(
                    "tool_call",
                    "search_uploaded_documents",
                    {"query": "opening time"},
                ),
                selection("final"),
            ],
        ):
            service.respond(
                question="When does the lab open?",
                history=[],
                system_prompt="You are Ollie.",
                use_documents=True,
            )

        self.assertEqual(
            captured["retrieved_documents"][0]["text"],
            "The guide says the lab opens at ten.",
        )
        self.assertEqual(observed_arguments, [{"query": "opening time"}])
        self.assertEqual(captured["tool_context"], [])


class BuiltInToolTests(unittest.TestCase):
    def test_datetime_tool_returns_timezone_aware_local_time(self) -> None:
        registry = ToolRegistry()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            registry = build_default_tool_registry(
                root / "ollie.db",
                root / "chroma",
                user_message="What time is it?",
                conversation_id="datetime-test",
            )
            result = ToolExecutor(registry).execute(
                "get_current_datetime",
                {},
                user_message="What time is it?",
            )
        self.assertTrue(result.success)
        local_time = datetime.fromisoformat(result.data["datetime"])
        self.assertIsNotNone(local_time.utcoffset())

    def test_document_search_reuses_rag_retrieval(self) -> None:
        match = {
            "text": "The manual says to restart the device.",
            "metadata": {"filename": "manual.pdf", "page_number": 2},
            "similarity": 0.91,
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            registry = build_default_tool_registry(
                root / "ollie.db",
                root / "chroma",
                user_message="Search my manual.",
                conversation_id="rag-tool",
            )
            with patch("agent.tools.search_documents", return_value=[match]) as search:
                result = ToolExecutor(registry).execute(
                    "search_uploaded_documents",
                    {"query": "restart"},
                    user_message="Search my manual.",
                )

        self.assertTrue(result.success)
        self.assertEqual(result.data["results"][0]["filename"], "manual.pdf")
        self.assertNotIn("id", result.data["results"][0])
        search.assert_called_once()

    def test_memory_search_and_explicit_memory_save_use_existing_service(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            database_path = root / "ollie.db"
            memory_service = MemoryService(MemoryStore(database_path))
            stored = memory_service.add_memory(
                "The user prefers Python for backend work.",
                category=MemoryCategory.PREFERENCES,
            )
            registry = build_default_tool_registry(
                database_path,
                root / "chroma",
                user_message="Remember that I prefer concise answers.",
                conversation_id="memory-tool",
            )
            executor = ToolExecutor(registry)
            retrieved = executor.execute(
                "retrieve_user_memories",
                {"query": "Python backend"},
                user_message="What language do I prefer for backend work?",
            )
            with patch(
                "memory.memory_updater.ollama.chat",
                return_value={
                    "message": {
                        "content": json.dumps(
                            {
                                "action": "add",
                                "target_id": None,
                                "confidence": 0.99,
                                "explanation": "It adds a separate preference.",
                            }
                        )
                    }
                },
            ):
                saved = executor.execute(
                    "save_user_memory",
                    {
                        "content": "The user prefers concise answers.",
                        "category": "preferences",
                        "importance": 0.8,
                    },
                    user_message="Remember that I prefer concise answers.",
                )
            records = memory_service.get_memories(category="preferences")

        self.assertTrue(retrieved.success)
        self.assertEqual(
            retrieved.data["memories"][0]["content"],
            stored.content,
        )
        self.assertTrue(saved.success)
        self.assertEqual(saved.data["status"], "created")
        self.assertEqual(len(records), 2)

    def test_qa_service_receives_registered_tool_results_separately(self) -> None:
        from rag.qa_service import answer_question

        with patch(
            "rag.qa_service.ollama.chat",
            return_value={"message": {"content": "The clock says noon."}},
        ) as chat:
            answer_question(
                "What time is it?",
                [],
                "You are Ollie.",
                use_documents=False,
                tool_context=['{"tool":"get_current_datetime","ok":true}'],
            )

        prompt = chat.call_args.kwargs["messages"]
        tool_message = next(
            message for message in prompt if "<tool_results>" in message["content"]
        )
        self.assertIn("get_current_datetime", tool_message["content"])

    def test_qa_reuses_agent_document_results_without_a_second_search(self) -> None:
        from rag.qa_service import answer_question

        documents = [
            {
                "text": "The guide says the lab opens at ten.",
                "document_id": None,
                "metadata": {"filename": "guide.pdf", "page_number": 2},
                "similarity": 0.9,
            }
        ]
        with patch("rag.qa_service.search_documents") as search:
            with patch(
                "rag.qa_service.ollama.chat",
                return_value={"message": {"content": "It opens at ten."}},
            ) as chat:
                answer = answer_question(
                    "When does the lab open?",
                    [],
                    "You are Ollie.",
                    use_documents=True,
                    retrieved_documents=documents,
                )

        search.assert_not_called()
        self.assertTrue(answer["used_documents"])
        self.assertEqual(answer["sources"][0]["filename"], "guide.pdf")
        document_message = next(
            message
            for message in chat.call_args.kwargs["messages"]
            if "<document_context>" in message["content"]
        )
        self.assertIn("opens at ten", document_message["content"])

    def test_use_documents_false_omits_document_search_tool(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            registry = build_default_tool_registry(
                root / "ollie.db",
                root / "chroma",
                user_message="Where is the uploaded document?",
                conversation_id="documents-disabled",
                include_document_search=False,
            )
        self.assertIsNone(registry.get("search_uploaded_documents"))


if __name__ == "__main__":
    unittest.main()
