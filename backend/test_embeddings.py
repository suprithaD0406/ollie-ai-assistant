"""Smoke-test single-text and batch embeddings through the local Ollama service."""

from rag.embedding_service import embed_text, embed_texts


EXPECTED_DIMENSIONS = 768


def main() -> None:
    """Generate embeddings and print basic success and vector details."""
    sample_text = "Ollie is a local-first personal AI assistant."
    embedding = embed_text(sample_text)
    assert len(embedding) == EXPECTED_DIMENSIONS, (
        f"Expected {EXPECTED_DIMENSIONS} dimensions, got {len(embedding)}."
    )

    print("Single embedding generation succeeded:", True)
    print("Vector dimensions:", len(embedding))
    print("First 10 vector values:", embedding[:10])

    sample_texts = [
        sample_text,
        "Embeddings represent text as numerical vectors.",
    ]
    embeddings = embed_texts(sample_texts)
    assert len(embeddings) == len(sample_texts), (
        f"Expected {len(sample_texts)} embeddings, got {len(embeddings)}."
    )
    assert all(len(vector) == EXPECTED_DIMENSIONS for vector in embeddings), (
        f"Every batch embedding should have {EXPECTED_DIMENSIONS} dimensions."
    )

    print("Batch embedding generation succeeded:", True)
    print("Batch vector dimensions:", [len(vector) for vector in embeddings])


if __name__ == "__main__":
    main()
