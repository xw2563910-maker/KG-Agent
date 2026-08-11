from typing import Any

import numpy as np

from embedding.model import (
    get_embedding_model,
)


DEFAULT_BATCH_SIZE = 32


def embed_chunks(
    chunks: list[dict[str, Any]],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """
    Convert PDF chunks into normalized
    document embeddings.
    """
    if not chunks:
        raise ValueError(
            "No chunks provided for embedding."
        )

    if batch_size < 1:
        raise ValueError(
            "batch_size must be at least 1."
        )

    texts = []

    for chunk in chunks:
        text = str(
            chunk.get("text") or ""
        ).strip()

        if not text:
            raise ValueError(
                "Chunk contains empty text: "
                f"{chunk.get('chunk_index')}"
            )

        texts.append(
            text
        )

    model = get_embedding_model()

    embeddings = model.encode_document(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return embeddings.astype(
        np.float32
    )


QUERY_INSTRUCTION = (
    "Represent this sentence for searching "
    "relevant passages: "
)


def embed_query(
    query: str,
) -> np.ndarray:
    """
    Convert a retrieval query into a normalized
    embedding vector.
    """
    query = query.strip()

    if not query:
        raise ValueError(
            "Query cannot be empty."
        )

    model = get_embedding_model()

    query_text = (
        QUERY_INSTRUCTION + query
    )

    embedding = model.encode(
        query_text,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embedding.astype(
        np.float32
    )