from typing import Any

import httpx

from config.settings import (
    OPENALEX_API_KEY,
    validate_openalex_config,
)


OPENALEX_WORKS_URL = "https://api.openalex.org/works"


def search_papers(
    query: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """
    Search academic papers from OpenAlex.

    Args:
        query: Search keywords.
        limit: Maximum number of papers to return.

    Returns:
        A list of normalized paper metadata dictionaries.
    """
    if not query.strip():
        raise ValueError("Paper search query cannot be empty.")

    if not 1 <= limit <= 25:
        raise ValueError("Paper search limit must be between 1 and 25.")

    validate_openalex_config()

    params = {
        "search": query,
        "per-page": limit,
        "api_key": OPENALEX_API_KEY,
    }

    try:
        response = httpx.get(
            OPENALEX_WORKS_URL,
            params=params,
            timeout=20.0,
        )

        response.raise_for_status()

    except httpx.HTTPError as exc:
        raise RuntimeError(
            f"OpenAlex request failed: {exc}"
        ) from exc

    data = response.json()

    papers = []

    for work in data.get("results", []):
        authors = []

        for authorship in work.get("authorships", []):
            author = authorship.get("author") or {}
            author_name = author.get("display_name")

            if author_name:
                authors.append(author_name)

        primary_location = work.get("primary_location") or {}
        source = primary_location.get("source") or {}

        paper = {
            "openalex_id": work.get("id"),
            "title": work.get("display_name"),
            "year": work.get("publication_year"),
            "doi": work.get("doi"),
            "type": work.get("type"),
            "language": work.get("language"),
            "cited_by_count": work.get("cited_by_count", 0),
            "authors": authors,
            "venue": source.get("display_name"),
        }

        papers.append(paper)

    return papers