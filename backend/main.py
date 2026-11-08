
from contextlib import contextmanager
import logging
import re
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi import BackgroundTasks
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool
import sqlite3
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Generator

from agent.agent_service import AgentService
from agent.tools import build_default_tool_registry
from memory.memory_models import MemoryInput
from memory.memory_extractor import extract_memories
from memory.memory_service import MemoryService
from memory.memory_store import MemoryStore
from rag.embedding_service import OllamaEmbeddingError
from rag.ingestion import DocumentIngestionError, InvalidPDFError, ingest_pdf
from rag.qa_service import QuestionAnsweringError, answer_question
from rag.vector_store import DEFAULT_PERSIST_DIRECTORY, VectorStoreError

app = FastAPI(title="Ollie AI Assistant")
logger = logging.getLogger(__name__)

# Database location
DB_PATH = Path(__file__).parent / "ollie.db"
CHROMA_DB_PATH = DEFAULT_PERSIST_DIRECTORY
DOCUMENT_UPLOAD_DIRECTORY = Path(__file__).parent / "storage" / "documents"
MAX_PDF_SIZE_BYTES = 20 * 1024 * 1024
MEMORY_TOP_K = 5
_CORRECTION_PATTERN = re.compile(
    r"\b(?:correction|actually|i meant|i now prefer|i no longer|instead)\b|"
    r"\bno\s*,",
    re.IGNORECASE,
)

SYSTEM_PROMPT = """
You are Ollie, a personal AI assistant.

IDENTITY:
- Your name is Ollie.
- You are the user's personal AI assistant.
- Never introduce yourself as Qwen.

RESPONSIBILITIES:
- Answer questions.
- Help organize tasks and activities.
- Assist with learning, coding and research.
- Help manage personal information.
- Help the user plan their work.

COMMUNICATION:
- Be friendly, natural and conversational.
- Be concise but informative.
- Understand previous conversation context.
- Avoid unnecessary introductions.
- Ask questions when instructions are unclear.

BEHAVIOUR:
- Be proactive and solution-oriented.
- Never invent information.
- Never claim to have performed an action
  unless you actually performed it.

Your purpose is to make the user's daily
work easier.
"""


# -----------------------------
# DATABASE INITIALIZATION
# -----------------------------

@contextmanager
def get_db_connection() -> Generator[sqlite3.Connection, None, None]:
    """Provide a transaction-scoped SQLite connection that always closes."""
    connection = sqlite3.connect(DB_PATH)
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def initialize_database():
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversation
            ON messages(conversation_id, id)
        """)


initialize_database()


# -----------------------------
# REQUEST MODEL
# -----------------------------

class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    conversation_id: str = Field(default="default", min_length=1)
    use_documents: bool | None = None

    @field_validator("message", "conversation_id")
    @classmethod
    def reject_blank_values(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("This field cannot be blank.")
        return value


# -----------------------------
# CHAT ENDPOINT
# -----------------------------

@app.post("/chat")
def chat(request: ChatRequest, background_tasks: BackgroundTasks):

    # Retrieve the latest 20 messages
    with get_db_connection() as conn:
        rows = conn.execute("""
            SELECT role, content
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id DESC
            LIMIT 20
        """, (request.conversation_id,)).fetchall()

    # Restore chronological order
    history = [
        {"role": role, "content": content}
        for role, content in reversed(rows)
    ]

    memory_context: list[str] = []
    memory_service: MemoryService | None = None
    try:
        memory_service = MemoryService(MemoryStore(DB_PATH))
        memory_context = [
            memory.content
            for memory in memory_service.retrieve_memories(
                request.message,
                top_k=MEMORY_TOP_K,
            )
        ]
    except Exception as exc:
        logger.warning(
            "Memory retrieval failed; continuing without memory context (%s).",
            type(exc).__name__,
        )

    try:
        tool_registry = build_default_tool_registry(
            DB_PATH,
            CHROMA_DB_PATH,
            user_message=request.message,
            conversation_id=request.conversation_id,
            memory_service=memory_service,
            include_document_search=request.use_documents is not False,
        )
        answer = AgentService(
            tool_registry,
            answerer=answer_question,
        ).respond(
            question=request.message,
            history=history,
            system_prompt=SYSTEM_PROMPT,
            use_documents=request.use_documents,
            persist_directory=CHROMA_DB_PATH,
            memory_context=memory_context,
        )
    except QuestionAnsweringError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    assistant_response = answer["response"]

    # Save both messages
    with get_db_connection() as conn:
        conn.executemany("""
            INSERT INTO messages
            (conversation_id, role, content)
            VALUES (?, ?, ?)
        """, [
            (
                request.conversation_id,
                "user",
                request.message
            ),
            (
                request.conversation_id,
                "assistant",
                assistant_response
            )
        ])

    background_tasks.add_task(
        extract_and_update_memories,
        request.message,
        request.conversation_id,
    )

    return {
        "assistant": "Ollie",
        "conversation_id": request.conversation_id,
        "response": assistant_response,
        "used_documents": answer["used_documents"],
        "sources": answer["sources"],
    }


def extract_and_update_memories(
    user_message: str,
    conversation_id: str,
) -> None:
    """Extract and reconcile memories from a user message without blocking its reply."""
    try:
        candidates = extract_memories(
            user_message,
            source_conversation_id=conversation_id,
        )
    except Exception as exc:
        logger.warning(
            "Memory extraction failed (%s).",
            type(exc).__name__,
        )
        return

    if not candidates:
        return

    try:
        memory_service = MemoryService(MemoryStore(DB_PATH))
    except Exception as exc:
        logger.warning(
            "Memory update failed because the memory store is unavailable (%s).",
            type(exc).__name__,
        )
        return

    explicit_correction = bool(_CORRECTION_PATTERN.search(user_message))
    for candidate in candidates:
        if not isinstance(candidate, MemoryInput):
            logger.warning("Memory extraction returned an invalid candidate type.")
            continue
        candidate = MemoryInput(
            content=candidate.content,
            category=candidate.category,
            importance=candidate.importance,
            source_conversation_id=conversation_id,
        )
        try:
            memory_service.apply_memory(
                candidate,
                explicit_correction=explicit_correction,
            )
        except Exception as exc:
            logger.warning(
                "Memory update failed (%s).",
                type(exc).__name__,
            )


@app.post("/documents")
async def upload_document(
    request: Request,
    filename: str = Query(..., min_length=1),
):
    """Ingest a raw PDF request body without requiring multipart support."""
    safe_filename = Path(filename.replace("\\", "/")).name
    if not safe_filename or safe_filename != filename or Path(safe_filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="A valid PDF filename is required.")

    content_type = request.headers.get("content-type", "").split(";", maxsplit=1)[0].strip()
    if content_type.lower() != "application/pdf":
        raise HTTPException(status_code=415, detail="Content-Type must be application/pdf.")

    DOCUMENT_UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            suffix=".pdf",
            prefix="ollie-upload-",
            dir=DOCUMENT_UPLOAD_DIRECTORY,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            total_bytes = 0
            pdf_signature = bytearray()
            async for data in request.stream():
                total_bytes += len(data)
                if total_bytes > MAX_PDF_SIZE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"PDF exceeds the {MAX_PDF_SIZE_BYTES // (1024 * 1024)} MB limit.",
                    )
                if len(pdf_signature) < 5:
                    pdf_signature.extend(data[: 5 - len(pdf_signature)])
                temporary_file.write(data)

        if total_bytes == 0:
            raise HTTPException(status_code=400, detail="The uploaded PDF is empty.")
        if pdf_signature != b"%PDF-":
            raise HTTPException(status_code=400, detail="The uploaded file is not a valid PDF.")

        result = await run_in_threadpool(
            ingest_pdf,
            temporary_path,
            safe_filename,
            persist_directory=CHROMA_DB_PATH,
        )
        return result
    except HTTPException:
        raise
    except (InvalidPDFError, FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (DocumentIngestionError, OllamaEmbeddingError, VectorStoreError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


# -----------------------------
# CONVERSATION HISTORY
# -----------------------------

@app.get("/history")
def get_history(conversation_id: str = "default"):

    with get_db_connection() as conn:
        rows = conn.execute("""
            SELECT role, content, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id ASC
        """, (conversation_id,)).fetchall()

    return {
        "conversation_id": conversation_id,
        "messages": [
            {
                "role": row[0],
                "content": row[1],
                "created_at": row[2]
            }
            for row in rows
        ]
    }


# -----------------------------
# BASIC ENDPOINTS
# -----------------------------

@app.get("/")
def home():
    return {"message": "Ollie AI Assistant is running!"}


@app.get("/health")
def health():
    return {"status": "ok"}
