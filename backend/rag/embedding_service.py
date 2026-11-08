"""Generate text embeddings with the locally running Ollama service."""

from collections.abc import Sequence
import math

import ollama


MODEL_NAME = "nomic-embed-text"


class OllamaEmbeddingError(RuntimeError):
    """Raised when Ollama cannot generate a valid embedding."""


def _validate_text(text: str) -> str:
    """Return a non-empty text value or raise a clear input error."""
    if not isinstance(text, str):
        raise TypeError("Each text to embed must be a string.")
    if not text.strip():
        raise ValueError("Text to embed cannot be empty.")
    return text


def embed_text(text: str) -> list[float]:
    """Generate an embedding vector for one text using Ollama."""
    return embed_texts([text])[0]


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """Generate one embedding vector for each text in a non-empty sequence."""
    if isinstance(texts, str) or not isinstance(texts, Sequence):
        raise TypeError("Texts to embed must be a sequence of strings.")
    if not texts:
        raise ValueError("At least one text is required.")

    validated_texts = [_validate_text(text) for text in texts]

    try:
        response = ollama.embed(model=MODEL_NAME, input=validated_texts)
    except (ollama.RequestError, ollama.ResponseError) as exc:
        raise OllamaEmbeddingError(
            f"Could not generate embeddings with Ollama model {MODEL_NAME!r}: {exc}"
        ) from exc

    embeddings = response.embeddings
    if len(embeddings) != len(validated_texts):
        raise OllamaEmbeddingError(
            "Ollama returned a different number of embeddings than requested."
        )

    vectors: list[list[float]] = []
    for embedding in embeddings:
        vector: list[float] = []
        for value in embedding:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise OllamaEmbeddingError("Ollama returned a non-numeric embedding value.")
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                raise OllamaEmbeddingError("Ollama returned a non-finite embedding value.")
            vector.append(numeric_value)

        if not vector:
            raise OllamaEmbeddingError("Ollama returned an empty embedding vector.")
        vectors.append(vector)

    return vectors