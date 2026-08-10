from pathlib import Path

from sentence_transformers import (
    SentenceTransformer,
    util,
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

QUERY_INSTRUCTION = (
    "Represent this sentence for searching "
    "relevant passages: "
)


def main():
    model = SentenceTransformer(
        str(MODEL_PATH),
        local_files_only=True,
    )

    query = (
        "empathetic response generation"
    )

    documents = [
        (
            "Empathetic Response Generation "
            "with Reinforcement Learning"
        ),
        (
            "Emotion Recognition "
            "in Dialogue Systems"
        ),
        (
            "Graph Neural Networks "
            "for Traffic Forecasting"
        ),
    ]

    query_text = (
        QUERY_INSTRUCTION
        + query
    )

    query_embedding = model.encode(
        query_text,
        convert_to_tensor=True,
    )

    document_embeddings = model.encode(
        documents,
        convert_to_tensor=True,
    )

    scores = util.cos_sim(
        query_embedding,
        document_embeddings,
    )[0]

    for document, score in zip(
        documents,
        scores,
    ):
        print(
            f"{float(score):.4f} "
            f"{document}"
        )


if __name__ == "__main__":
    main()