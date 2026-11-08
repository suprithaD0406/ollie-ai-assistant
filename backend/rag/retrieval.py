"""Semantic search over Ollie's persisted document chunks."""

from pathlib import Path
import sqlite3

import chromadb
from chromadb.errors import ChromaError, NotFoundError

from rag.embedding_service import embed_text
from rag.vector_store import COLLECTION_NAME, DEFAULT_PERSIST_DIRECTORY


class RetrievalError(RuntimeError):
    """Raised when the document collection cannot be searched."""


class CollectionNotFoundError(RetrievalError):
    """Raised when the persistent document collection has not been initialized."""


def search_documents(
    query: str,
    top_k: int = 5,
    persist_directory: str | Path | None = None,
) -> list[dict[str, object]]:
    """Return the most similar stored chunks for a natural-language query.

    ChromaDB's cosine distance is returned as ``distance``. ``similarity`` is
    calculated as ``1 - distance`` for convenient ranking interpretation.
    """
    if not isinstance(query, str):
        raise TypeError("Query must be a string.")
    if not query.strip():
        raise ValueError("Query cannot be empty.")
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k must be a positive integer.")
    if top_k < 1:
        raise ValueError("top_k must be a positive integer.")
    if persist_directory is not None and not isinstance(persist_directory, (str, Path)):
        raise TypeError("persist_directory must be a string or Path.")
    if isinstance(persist_directory, str) and not persist_directory.strip():
        raise ValueError("persist_directory cannot be empty.")

    database_path = Path(
        DEFAULT_PERSIST_DIRECTORY if persist_directory is None else persist_directory
    )
    client = None
    try:
        client = chromadb.PersistentClient(path=str(database_path))
        try:
            collection = client.get_collection(
                name=COLLECTION_NAME,
                embedding_function=None,
            )
        except NotFoundError as exc:
            raise CollectionNotFoundError(
                f"ChromaDB collection {COLLECTION_NAME!r} was not found at {database_path}."
            ) from exc

        if not collection.metadata or collection.metadata.get("hnsw:space") != "cosine":
            raise RetrievalError(
                f"ChromaDB collection {COLLECTION_NAME!r} is not configured for cosine distance."
            )

        if collection.count() == 0:
            return []

        query_embedding = embed_text(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
    except RetrievalError:
        raise
    except (ChromaError, OSError, sqlite3.Error) as exc:
        raise RetrievalError(f"Could not search the ChromaDB collection: {exc}") from exc
    finally:
        if client is not None:
            client.close()

    ids_by_query = results.get("ids") or [[]]
    documents_by_query = results.get("documents") or [[]]
    metadatas_by_query = results.get("metadatas") or [[]]
    distances_by_query = results.get("distances") or [[]]
    result_ids = ids_by_query[0] if ids_by_query else []
    documents = documents_by_query[0] if documents_by_query else []
    metadatas = metadatas_by_query[0] if metadatas_by_query else []
    distances = distances_by_query[0] if distances_by_query else []

    matches: list[dict[str, object]] = []
    for index, chunk_id in enumerate(result_ids):
        distance = distances[index] if index < len(distances) else None
        metadata = metadatas[index] if index < len(metadatas) else None
        matches.append(
            {
                "id": chunk_id,
                "text": documents[index] if index < len(documents) else None,
                "document_id": metadata.get("document_id") if metadata else None,
                "metadata": metadata,
                "distance": distance,
                "similarity": 1.0 - distance if distance is not None else None,
            }
        )

    return matches