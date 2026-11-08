"""Persistent ChromaDB storage for Ollie's document chunks."""

from collections.abc import Mapping, Sequence
import math
from pathlib import Path
import sqlite3
from types import TracebackType
import uuid

import chromadb
from chromadb.errors import ChromaError, DuplicateIDError, IDAlreadyExistsError

from rag.embedding_service import embed_texts


COLLECTION_NAME = "ollie_documents"
DEFAULT_PERSIST_DIRECTORY = Path(__file__).parent.parent / "storage" / "chroma_db"


class VectorStoreError(RuntimeError):
    """Raised when the persistent vector store cannot complete an operation."""


class DuplicateChunkError(VectorStoreError):
    """Raised when a generated chunk ID already exists in the collection."""


class VectorStore:
    """Manage Ollie's persistent ChromaDB collection and document chunks."""

    def __init__(self, persist_directory: str | Path | None = None) -> None:
        """Open the persistent collection, creating it with cosine distance if needed."""
        if persist_directory is not None and not isinstance(persist_directory, (str, Path)):
            raise TypeError("persist_directory must be a string or Path.")
        if isinstance(persist_directory, str) and not persist_directory.strip():
            raise ValueError("persist_directory cannot be empty.")
        self.persist_directory = Path(
            DEFAULT_PERSIST_DIRECTORY if persist_directory is None else persist_directory
        )
        self._closed = False

        client = None
        try:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(self.persist_directory))
            collection = client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
                embedding_function=None,
            )
        except (ChromaError, OSError, sqlite3.Error) as exc:
            if client is not None:
                client.close()
            raise VectorStoreError(
                f"Could not initialize ChromaDB at {self.persist_directory}: {exc}"
            ) from exc

        if not collection.metadata or collection.metadata.get("hnsw:space") != "cosine":
            client.close()
            raise VectorStoreError(
                f"Collection {COLLECTION_NAME!r} is not configured for cosine distance."
            )

        self._client = client
        self.collection = collection

    def add_chunks(
        self,
        chunks: Sequence[Mapping[str, object]],
        document_id: str,
        filename: str,
    ) -> list[str]:
        """Embed and store chunks, returning their generated unique IDs."""
        self._ensure_open()
        if isinstance(chunks, (str, bytes)) or not isinstance(chunks, Sequence):
            raise TypeError("Chunks must be a sequence of mappings.")
        if not chunks:
            raise ValueError("At least one chunk is required.")
        self._validate_label(document_id, "document_id")
        self._validate_label(filename, "filename")

        texts: list[str] = []
        page_numbers: list[int] = []
        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, Mapping):
                raise TypeError(f"Chunk {index} must be a mapping.")

            text = chunk.get("text")
            page_number = chunk.get("page_number")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Chunk {index} must contain non-empty text.")
            if (
                isinstance(page_number, bool)
                or not isinstance(page_number, int)
                or page_number < 1
            ):
                raise ValueError(f"Chunk {index} must have a positive integer page_number.")

            texts.append(text)
            page_numbers.append(page_number)

        embeddings = embed_texts(texts)
        self._validate_embeddings(embeddings, len(texts))

        ids = [uuid.uuid4().hex for _ in texts]
        if len(set(ids)) != len(ids):
            ids = [f"{uuid.uuid4().hex}-{index}" for index in range(len(texts))]
        if len(set(ids)) != len(ids):
            raise DuplicateChunkError("Could not generate unique IDs for all document chunks.")

        metadatas = [
            {
                "document_id": document_id,
                "filename": filename,
                "page_number": page_number,
                "chunk_index": index,
            }
            for index, page_number in enumerate(page_numbers)
        ]

        try:
            existing_ids = self.collection.get(ids=ids, include=[])["ids"]
            if existing_ids:
                raise DuplicateChunkError(
                    f"Generated chunk ID already exists: {existing_ids[0]}"
                )
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
        except (DuplicateIDError, IDAlreadyExistsError) as exc:
            raise DuplicateChunkError(
                "ChromaDB rejected a chunk because its ID already exists."
            ) from exc
        except (ChromaError, OSError, sqlite3.Error) as exc:
            raise VectorStoreError(f"Could not store document chunks: {exc}") from exc

        return ids

    def get_chunk_count(self) -> int:
        """Return the number of chunks currently stored in the collection."""
        self._ensure_open()
        try:
            return self.collection.count()
        except (ChromaError, OSError, sqlite3.Error) as exc:
            raise VectorStoreError(f"Could not count stored document chunks: {exc}") from exc

    def close(self) -> None:
        """Close the ChromaDB client and release its resources."""
        if not self._closed:
            try:
                self._client.close()
            except (ChromaError, OSError, sqlite3.Error) as exc:
                raise VectorStoreError(f"Could not close the ChromaDB client: {exc}") from exc
            self._closed = True

    def __enter__(self) -> "VectorStore":
        """Return this store for use as a context manager."""
        self._ensure_open()
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        """Close the client when leaving a context manager."""
        self.close()

    @staticmethod
    def _validate_label(value: str, name: str) -> None:
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string.")
        if not value.strip():
            raise ValueError(f"{name} cannot be empty.")

    @staticmethod
    def _validate_embeddings(embeddings: list[list[float]], expected_count: int) -> None:
        if len(embeddings) != expected_count:
            raise VectorStoreError(
                "The embedding service returned a different number of vectors than requested."
            )
        if not embeddings or not embeddings[0]:
            raise VectorStoreError("The embedding service returned an empty vector.")

        dimensions = len(embeddings[0])
        for vector in embeddings:
            if len(vector) != dimensions or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                for value in vector
            ):
                raise VectorStoreError(
                    "The embedding service returned invalid or inconsistent vectors."
                )

    def _ensure_open(self) -> None:
        if self._closed:
            raise VectorStoreError("The vector store is closed.")


def add_chunks(
    chunks: Sequence[Mapping[str, object]],
    document_id: str,
    filename: str,
    persist_directory: str | Path | None = None,
) -> list[str]:
    """Persist document chunks in the default or specified ChromaDB directory."""
    with VectorStore(persist_directory) as store:
        return store.add_chunks(chunks, document_id, filename)


def get_chunk_count(persist_directory: str | Path | None = None) -> int:
    """Return the number of stored chunks in the default or specified database."""
    with VectorStore(persist_directory) as store:
        return store.get_chunk_count()