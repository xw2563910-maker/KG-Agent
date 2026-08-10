from typing import Any


DEFAULT_MIN_ABSTRACT_WORDS = 50


def normalize_doi(doi: str) -> str:
    """
    Normalize a DOI string so that different DOI formats
    can be compared consistently.
    """
    normalized = doi.strip().lower()

    prefixes = (
        "https://doi.org/",
        "http://doi.org/",
        "doi:",
    )

    for prefix in prefixes:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
            break

    return normalized


def get_paper_identity(
    paper: dict[str, Any],
) -> str:
    """
    Build a stable identity for paper deduplication.

    Priority:
        1. DOI
        2. OpenAlex ID
        3. Normalized title
    """
    doi = paper.get("doi")

    if doi:
        normalized_doi = normalize_doi(
            str(doi)
        )

        return f"doi:{normalized_doi}"

    openalex_id = paper.get("openalex_id")

    if openalex_id:
        normalized_openalex_id = (
            str(openalex_id)
            .strip()
            .lower()
        )

        return (
            f"openalex:{normalized_openalex_id}"
        )

    title = str(
        paper.get("title") or ""
    )

    normalized_title = " ".join(
        title.lower().split()
    )

    return f"title:{normalized_title}"


def evaluate_paper_quality(
    paper: dict[str, Any],
    min_abstract_words: int = DEFAULT_MIN_ABSTRACT_WORDS,
) -> tuple[bool, str]:
    """
    Evaluate whether a paper is suitable
    to be used as academic evidence.

    Returns:
        A tuple:

        (
            is_valid,
            reason,
        )
    """
    title = str(
        paper.get("title") or ""
    ).strip()

    if not title:
        return False, "missing title"

    abstract = str(
        paper.get("abstract") or ""
    ).strip()

    if not abstract:
        return False, "missing abstract"

    word_count = len(
        abstract.split()
    )

    if word_count < min_abstract_words:
        return (
            False,
            f"abstract too short: {word_count} words",
        )

    return True, "accepted"


def filter_papers(
    papers: list[dict[str, Any]],
    min_abstract_words: int = DEFAULT_MIN_ABSTRACT_WORDS,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, str]],
]:
    """
    Filter duplicate and low-quality papers.

    Returns:
        selected_papers:
            Papers accepted as evidence.

        rejected_papers:
            Papers rejected with reasons.
    """

    selected_papers = []
    rejected_papers = []

    seen_papers = set()

    for paper in papers:
        title = str(
            paper.get("title")
            or "Unknown title"
        )

        identity = get_paper_identity(
            paper
        )

        if identity in seen_papers:
            rejected_papers.append(
                {
                    "title": title,
                    "reason": "duplicate paper",
                }
            )

            continue

        is_valid, reason = evaluate_paper_quality(
            paper,
            min_abstract_words=min_abstract_words,
        )

        if not is_valid:
            rejected_papers.append(
                {
                    "title": title,
                    "reason": reason,
                }
            )

            continue

        seen_papers.add(identity)

        selected_papers.append(
            paper
        )


    return selected_papers, rejected_papers