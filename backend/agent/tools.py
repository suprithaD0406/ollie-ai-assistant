"""Initial local tools backed by Ollie's existing RAG and memory services."""

from datetime import datetime
import logging
from pathlib import Path
from typing import Any

from agent.tool_registry import ToolRegistry, ToolSpec
from memory.memory_extractor import contains_sensitive_data
from memory.memory_models import MemoryInput
from memory.memory_service import MemoryService
from memory.memory_store import MemoryStore, MemoryStoreError
from rag.retrieval import CollectionNotFoundError, search_documents


_TOP_K_SCHEMA: dict[str, Any] = {
    "type": "integer",
    "minimum": 1,
    "maximum": 5,
    "default": 5,
}
logger = logging.getLogger(__name__)


def build_default_tool_registry(
    database_path: str | Path,
    persist_directory: str | Path,
    *,
    user_message: str,
    conversation_id: str,
    memory_service: MemoryService | None = None,
    include_document_search: bool = True,
) -> ToolRegistry:
    """Create the four explicitly supported tools for one chat request."""
    if memory_service is None:
        try:
            memory_service = MemoryService(MemoryStore(database_path))
        except MemoryStoreError as exc:
            logger.warning(
                "Memory tools are unavailable (%s).",
                type(exc).__name__,
            )
    registry = ToolRegistry()

    registry.register(
        ToolSpec(
            name="get_current_datetime",
            description="Get the current local date and time with its timezone offset.",
            input_schema={
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            handler=_current_datetime,
        )
    )
    if include_document_search:
        registry.register(
            ToolSpec(
                name="search_uploaded_documents",
                description=(
                    "Search locally uploaded documents for excerpts relevant to a query. "
                    "Use only for questions about uploaded files."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "maxLength": 1000},
                        "top_k": _TOP_K_SCHEMA,
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                handler=_document_search_handler(persist_directory),
            )
        )
    if memory_service is not None:
        registry.register(
            ToolSpec(
                name="retrieve_user_memories",
                description=(
                    "Retrieve relevant active personal memories. Use only when the "
                    "conversation needs known user-specific information."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "maxLength": 1000},
                        "top_k": _TOP_K_SCHEMA,
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                handler=_memory_search_handler(memory_service),
            )
        )
        registry.register(
            ToolSpec(
                name="save_user_memory",
                description=(
                    "Save or update a durable fact about the user only when the latest "
                    "user message explicitly asks Ollie to remember, save, or store it."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "maxLength": 1000},
                        "category": {
                            "type": "string",
                            "enum": [
                                "personal",
                                "education",
                                "career",
                                "preferences",
                                "projects",
                                "goals",
                                "other",
                            ],
                        },
                        "importance": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                    },
                    "required": ["content", "category", "importance"],
                    "additionalProperties": False,
                },
                handler=_memory_save_handler(
                    memory_service,
                    conversation_id,
                    user_message,
                ),
            )
        )
    return registry


def _document_search_handler(persist_directory: str | Path):
    def search(arguments: dict[str, Any]) -> dict[str, object]:
        try:
            matches = search_documents(
                arguments["query"],
                top_k=arguments.get("top_k", 5),
                persist_directory=persist_directory,
            )
        except CollectionNotFoundError:
            matches = []
        results = []
        for match in matches:
            metadata = match.get("metadata")
            metadata = metadata if isinstance(metadata, dict) else {}
            text = match.get("text")
            results.append(
                {
                    "text": text[:4000] if isinstance(text, str) else "",
                    "filename": metadata.get("filename"),
                    "page_number": metadata.get("page_number"),
                    "similarity": match.get("similarity"),
                }
            )
        return {"results": results}

    return search


def _current_datetime(arguments: dict[str, Any]) -> dict[str, str]:
    if arguments:
        raise ValueError("get_current_datetime does not accept arguments.")
    return {"datetime": datetime.now().astimezone().isoformat()}


def _memory_search_handler(memory_service: MemoryService):
    def search(arguments: dict[str, Any]) -> dict[str, object]:
        memories = memory_service.retrieve_memories(
            arguments["query"],
            top_k=arguments.get("top_k", 5),
        )
        return {
            "memories": [
                {"content": memory.content, "category": memory.category.value}
                for memory in memories
            ]
        }

    return search


def _memory_save_handler(
    memory_service: MemoryService,
    conversation_id: str,
    user_message: str,
):
    def save(arguments: dict[str, Any]) -> dict[str, object]:
        if contains_sensitive_data(arguments["content"]):
            raise ValueError("Sensitive information cannot be saved as a memory.")
        candidate = MemoryInput(
            content=arguments["content"],
            category=arguments["category"],
            importance=arguments["importance"],
            source_conversation_id=conversation_id,
        )
        result = memory_service.apply_memory(
            candidate,
            explicit_correction=_looks_like_correction(user_message),
        )
        return {"status": result.action.value}

    return save


def _looks_like_correction(text: str) -> bool:
    lowered = text.casefold()
    return any(
        phrase in lowered
        for phrase in ("correction", "actually", "i meant", "instead", "no,")
    )
