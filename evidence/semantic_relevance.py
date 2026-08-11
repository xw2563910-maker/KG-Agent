from pathlib import Path
from typing import Any

from sentence_transformers import (
    SentenceTransformer,
    util,
)
from embedding.model import (
    get_embedding_model,
)


QUERY_INSTRUCTION = (
    "Represent this sentence for searching "
    "relevant passages: "
)


def build_paper_text(
    paper: dict[str, Any],
) -> str:
    """
    Build the document text used for
    semantic relevance ranking.
    """
    title = str(
        paper.get("title") or ""
    ).strip()

    abstract = str(
        paper.get("abstract") or ""
    ).strip()

    return (
        f"{title}\n\n"
        f"{abstract}"
    ).strip()


def compute_semantic_scores(
    query: str,
    papers: list[dict[str, Any]],
) -> list[float]:
    """
    Compute cosine similarity scores between
    the search query and all papers.
    """
    if not query.strip():
        raise ValueError(
            "Semantic ranking query cannot be empty."
        )

    if not papers:
        return []

    model = get_embedding_model()

    query_text = (
        QUERY_INSTRUCTION
        + query.strip()
    )

    document_texts = [
        build_paper_text(paper)
        for paper in papers
    ]

    query_embedding = model.encode(
        query_text,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    document_embeddings = model.encode(
        document_texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    scores = util.cos_sim(
        query_embedding,
        document_embeddings,
    )[0]

    return [
        round(float(score), 4)
        for score in scores
    ]


def rank_papers_semantically(
    papers: list[dict[str, Any]],
    query: str,
) -> list[dict[str, Any]]:
    """
    Rank papers from most semantically relevant
    to least semantically relevant.
    """
    scores = compute_semantic_scores(
        query,
        papers,
    )

    ranked_papers = []

    for paper, score in zip(
        papers,
        scores,
    ):
        ranked_paper = {
            **paper,
            "semantic_score": score,
        }

        ranked_papers.append(
            ranked_paper
        )

    ranked_papers.sort(
        key=lambda paper: paper[
            "semantic_score"
        ],
        reverse=True,
    )

    return ranked_papers