from typing import Any

import faiss
import numpy as np


class VectorStore:
    def __init__(
        self,
        chunks: list[dict[str, Any]],
        embeddings: np.ndarray,
    ):
        if not chunks:
            raise ValueError(
                "Chunks cannot be empty."
            )

        if embeddings.ndim != 2:
            raise ValueError(
                "Embeddings must be a "
                "2-dimensional matrix."
            )

        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                "Number of chunks must match "
                "number of embeddings."
            )

        self.chunks = chunks

        self.embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        dimension = self.embeddings.shape[1]

        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.index.add(
            self.embeddings
        )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        if top_k < 1:
            raise ValueError(
                "top_k must be at least 1."
            )

        query = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        if query.ndim == 1:
            query = query.reshape(
                1,
                -1,
            )

        if query.ndim != 2:
            raise ValueError(
                "Query embedding must be "
                "1D or 2D."
            )

        if query.shape[1] != self.index.d:
            raise ValueError(
                "Query embedding dimension "
                "does not match index dimension."
            )

        top_k = min(
            top_k,
            self.index.ntotal,
        )

        scores, indices = self.index.search(
            query,
            top_k,
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0],
        ):
            result = {
                **self.chunks[int(index)],
                "score": round(
                    float(score),
                    4,
                ),
            }

            results.append(
                result
            )

        return results