"""PDF ingestion orchestration for Ollie's document collection."""

from pathlib import Path
import uuid

import fitz

from rag.chunking import chunk_documents
from rag.document_loader import load_pdf
from rag.embedding_service import OllamaEmbeddingError
from rag.vector_store import DEFAULT_PERSIST_DIRECTORY, VectorStoreError, add_chunks


class InvalidPDFError(ValueError):
    """Raised when a PDF cannot be read or contains no extractable text."""


class DocumentIngestionError(RuntimeError):
    """Raised when a PDF could not be fully ingested."""


def ingest_pdf(
    file_path: str | Path,
    filename: str | None = None,
    document_id: str | None = None,
    persist_directory: str | Path | None = None,
) -> dict[str, object]:
    """Extract, chunk, embed, and persist one PDF using the existing RAG modules."""
    pdf_path = Path(file_path)
    if pdf_path.suffix.lower() != ".pdf":
        raise InvalidPDFError("Only PDF files are supported.")

    safe_filename = filename if filename is not None else pdf_path.name
    if not isinstance(safe_filename, str) or not safe_filename.strip():
        raise ValueError("filename must be a non-empty string.")
    if Path(safe_filename.replace("\\", "/")).name != safe_filename:
        raise ValueError("filename must not contain a directory path.")
    if Path(safe_filename).suffix.lower() != ".pdf":
        raise InvalidPDFError("Only PDF files are supported.")

    if document_id is None:
        document_id = uuid.uuid4().hex
    elif not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("document_id must be a non-empty string.")

    try:
        with fitz.open(stream=pdf_path.read_bytes(), filetype="pdf"):
            pass
    except (fitz.FileDataError, fitz.EmptyFileError) as exc:
        raise InvalidPDFError(f"Could not read the uploaded PDF: {exc}") from exc

    try:
        pages = load_pdf(str(pdf_path))
    except (fitz.FileDataError, fitz.EmptyFileError) as exc:
        raise InvalidPDFError(f"Could not read the uploaded PDF: {exc}") from exc
    except ValueError as exc:
        raise InvalidPDFError(str(exc)) from exc
    except FileNotFoundError:
        raise

    chunks = chunk_documents(pages)
    if not chunks:
        raise InvalidPDFError("The PDF did not contain any text chunks to store.")

    try:
        chunk_ids = add_chunks(
            chunks,
            document_id=document_id,
            filename=safe_filename,
            persist_directory=(
                DEFAULT_PERSIST_DIRECTORY if persist_directory is None else persist_directory
            ),
        )
    except (OllamaEmbeddingError, VectorStoreError) as exc:
        raise DocumentIngestionError(f"Could not store PDF chunks: {exc}") from exc

    return {
        "document_id": document_id,
        "filename": safe_filename,
        "page_count": len(pages),
        "chunk_count": len(chunk_ids),
        "chunk_ids": chunk_ids,
    }
