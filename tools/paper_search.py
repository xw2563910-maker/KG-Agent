from typing import Any

import httpx

from config.settings import (
    OPENALEX_API_KEY,
    validate_openalex_config,
)


OPENALEX_WORKS_URL = "https://api.openalex.org/works"


def reconstruct_abstract(
    inverted_index: dict[str, list[int]] | None,
) -> str | None:
    """
    Reconstruct an abstract from OpenAlex's inverted index format.

    Example:
        {
            "Knowledge": [0],
            "graphs": [1],
            "are": [2],
            "useful": [3]
        }

    Becomes:
        "Knowledge graphs are useful"
    """
    if not inverted_index:
        return None

    positions = [
        position
        for word_positions in inverted_index.values()
        for position in word_positions
    ]

    if not positions:
        return None

    words = [""] * (max(positions) + 1)

    for word, word_positions in inverted_index.items():
        for position in word_positions:
            if 0 <= position < len(words):
                words[position] = word

    abstract = " ".join(
        word
        for word in words
        if word
    )

    return abstract or None


def search_papers(
    query: str,
    limit: int = 5,
    from_year: int | None = None,
    to_year: int | None = None,
    require_abstract: bool = False,
) -> list[dict[str, Any]]:
    """
    Search academic papers from OpenAlex.

    Args:
        query:
            Academic search keywords.

        limit:
            Maximum number of papers to return.

        from_year:
            Beginning publication year.

        to_year:
            Ending publication year.

        require_abstract:
            Whether returned papers must contain abstracts.

    Returns:
        A list of normalized paper metadata dictionaries.
    """
    if not query.strip():
        raise ValueError(
            "Paper search query cannot be empty."
        )

    if not 1 <= limit <= 25:
        raise ValueError(
            "Paper search limit must be between 1 and 25."
        )

    if (from_year is None) != (to_year is None):
        raise ValueError(
            "from_year and to_year must be provided together."
        )

    if (
        from_year is not None
        and to_year is not None
        and from_year > to_year
    ):
        raise ValueError(
            "from_year cannot be greater than to_year."
        )

    validate_openalex_config()

    filters = []

    if from_year is not None and to_year is not None:
        filters.append(
            f"publication_year:{from_year}-{to_year}"
        )

    if require_abstract:
        filters.append(
            "has_abstract:true"
        )

    params = {
        "search": query,
        "per-page": limit,
        "api_key": OPENALEX_API_KEY,
    }

    if filters:
        params["filter"] = ",".join(filters)

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
        author_entities = []

        for authorship in work.get(
            "authorships",
            [],
        ):
            author = (
                authorship.get("author")
                or {}
            )

            author_id = author.get(
                "id"
            )

            author_name = author.get(
                "display_name"
            )

            # Keep the original author-name list
            # for backward compatibility.
            if author_name:
                authors.append(
                    author_name
                )

            # Keep structured OpenAlex author
            # metadata for the knowledge graph.
            if author_id:
                author_entities.append(
                    {
                        "openalex_id": author_id,
                        "name": author_name,
                    }
                )

        primary_location = (
            work.get("primary_location")
            or {}
        )

        source = (
            primary_location.get("source")
            or {}
        )

        venue_id = source.get(
            "id"
        )

        venue_name = source.get(
            "display_name"
        )

        venue_entity = None

        if venue_id:
            venue_entity = {
                "openalex_id": venue_id,
                "name": venue_name,
            }

        abstract = reconstruct_abstract(
            work.get(
                "abstract_inverted_index"
            )
        )

        paper = {
            "openalex_id": work.get("id"),

            "title": work.get(
                "display_name"
            ),

            "year": work.get(
                "publication_year"
            ),

            "publication_date": work.get(
                "publication_date"
            ),

            "doi": work.get(
                "doi"
            ),

            "type": work.get(
                "type"
            ),

            "language": work.get(
                "language"
            ),

            "cited_by_count": work.get(
                "cited_by_count",
                0,
            ),

            # Existing fields used by the
            # current research pipeline.
            "authors": authors,
            "venue": venue_name,

            # New structured fields used
            # by the knowledge graph.
            "author_entities": (
                author_entities
            ),

            "venue_entity": (
                venue_entity
            ),

            "abstract": abstract,
        }

        papers.append(
            paper
        )

    return papers