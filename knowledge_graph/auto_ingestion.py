from typing import Any

from neo4j import Driver

from knowledge_graph.entity_extraction import (
    extract_scientific_entities,
)
from knowledge_graph.ingestion import (
    ingest_paper,
)
from knowledge_graph.semantic_ingestion import (
    ingest_scientific_entities,
)


def ingest_research_papers(
    driver: Driver,
    papers: list[dict[str, Any]],
) -> dict[str, Any]:
    if not isinstance(
        papers,
        list,
    ):
        raise TypeError(
            "Papers must be a list."
        )

    report: dict[str, Any] = {
        "total": len(papers),
        "metadata_ingested": 0,
        "semantic_ingested": 0,
        "skipped": 0,
        "errors": [],
    }

    for paper in papers:
        if not isinstance(
            paper,
            dict,
        ):
            report[
                "skipped"
            ] += 1

            report[
                "errors"
            ].append(
                {
                    "paper_id": None,
                    "title": None,
                    "stage": "validation",
                    "error": (
                        "Paper must be "
                        "a dictionary."
                    ),
                }
            )

            continue

        paper_id = str(
            paper.get(
                "openalex_id"
            )
            or ""
        ).strip()

        title = str(
            paper.get(
                "title"
            )
            or ""
        ).strip()

        if not paper_id:
            report[
                "skipped"
            ] += 1

            report[
                "errors"
            ].append(
                {
                    "paper_id": None,
                    "title": title,
                    "stage": "validation",
                    "error": (
                        "Missing Paper "
                        "OpenAlex ID."
                    ),
                }
            )

            continue

        # ========================================================
        # Stage 1:
        # Deterministic OpenAlex Metadata
        # ========================================================

        try:
            ingest_paper(
                driver,
                paper,
            )

            report[
                "metadata_ingested"
            ] += 1

        except Exception as exc:
            report[
                "errors"
            ].append(
                {
                    "paper_id": paper_id,
                    "title": title,
                    "stage": "metadata",
                    "error": str(exc),
                }
            )

            # Semantic entities depend on
            # the Paper node existing.
            continue

        # ========================================================
        # Stage 2:
        # LLM Scientific Entity Extraction
        # ========================================================

        try:
            entities = (
                extract_scientific_entities(
                    paper
                )
            )

            ingest_scientific_entities(
                driver,
                paper_id,
                entities,
            )

            report[
                "semantic_ingested"
            ] += 1

        except Exception as exc:
            report[
                "errors"
            ].append(
                {
                    "paper_id": paper_id,
                    "title": title,
                    "stage": "semantic",
                    "error": str(exc),
                }
            )

    return report