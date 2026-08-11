from pathlib import Path

from sentence_transformers import (
    SentenceTransformer,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "embedding"
    / "bge-small-en-v1.5"
)


_embedding_model = None


def get_embedding_model() -> SentenceTransformer:
    """
    Load and cache the local embedding model.
    """
    global _embedding_model

    if _embedding_model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Embedding model not found: "
                f"{MODEL_PATH}"
            )

        _embedding_model = SentenceTransformer(
            str(MODEL_PATH),
            local_files_only=True,
        )

    return _embedding_model