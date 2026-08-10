import re
from typing import Any


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize_text(
    text: str,
) -> set[str]:
    """
    Normalize text and convert it into
    a set of lowercase tokens.
    """
    return set(
        TOKEN_PATTERN.findall(
            text.lower()
        )
    )


def compute_relevance_score(
    query: str,
    paper: dict[str, Any],
) -> float:
    """
    Compute a simple lexical relevance score
    between a search query and a paper.

    Title matches receive a higher weight
    than abstract matches.
    """
    query_tokens = tokenize_text(
        query
    )

    if not query_tokens:
        return 0.0

    title = str(
        paper.get("title") or ""
    )

    abstract = str(
        paper.get("abstract") or ""
    )

    title_tokens = tokenize_text(
        title
    )

    abstract_tokens = tokenize_text(
        abstract
    )

    title_matches = (
        query_tokens
        & title_tokens
    )

    abstract_matches = (
        query_tokens
        & abstract_tokens
    )

    title_coverage = (
        len(title_matches)
        / len(query_tokens)
    )

    abstract_coverage = (
        len(abstract_matches)
        / len(query_tokens)
    )

    score = (
        0.7 * title_coverage
        + 0.3 * abstract_coverage
    )

    return round(
        score,
        4,
    )


def rank_papers(
    papers: list[dict[str, Any]],
    query: str,
) -> list[dict[str, Any]]:
    """
    Rank papers from most relevant
    to least relevant.

    Each returned paper contains
    a relevance_score field.
    """
    if not query.strip():
        raise ValueError(
            "Relevance ranking query cannot be empty."
        )

    ranked_papers = []

    for paper in papers:
        score = compute_relevance_score(
            query,
            paper,
        )

        ranked_paper = {
            **paper,
            "relevance_score": score,
        }

        ranked_papers.append(
            ranked_paper
        )

    ranked_papers.sort(
        key=lambda paper: paper[
            "relevance_score"
        ],
        reverse=True,
    )

    return ranked_papers