from typing import Any

from neo4j import Driver

from knowledge_graph.query import (
    find_papers_by_dataset,
    find_papers_by_method,
    find_papers_by_task,
    find_papers_by_title,
    get_paper_knowledge,
)


def retrieve_kg(
    driver: Driver,
    plan: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(plan, dict):
        raise TypeError(
            "KG query plan must be a dictionary."
        )

    entity_type = str(
        plan.get(
            "entity_type",
            "",
        )
    ).strip().lower()

    entity_value = str(
        plan.get(
            "entity_value",
            "",
        )
    ).strip()

    valid_entity_types = {
        "paper",
        "method",
        "dataset",
        "task",
    }

    if entity_type not in valid_entity_types:
        raise ValueError(
            "Unsupported KG entity type: "
            f"{entity_type}"
        )

    if not entity_value:
        raise ValueError(
            "KG entity value cannot be empty."
        )

    if entity_type == "method":
        results = find_papers_by_method(
            driver,
            entity_value,
        )

        return {
            "entity_type": entity_type,
            "entity_value": entity_value,
            "status": (
                "ok"
                if results
                else "not_found"
            ),
            "results": results,
        }

    if entity_type == "dataset":
        results = find_papers_by_dataset(
            driver,
            entity_value,
        )

        return {
            "entity_type": entity_type,
            "entity_value": entity_value,
            "status": (
                "ok"
                if results
                else "not_found"
            ),
            "results": results,
        }

    if entity_type == "task":
        results = find_papers_by_task(
            driver,
            entity_value,
        )

        return {
            "entity_type": entity_type,
            "entity_value": entity_value,
            "status": (
                "ok"
                if results
                else "not_found"
            ),
            "results": results,
        }

    return _retrieve_paper(
        driver,
        entity_value,
    )


def _retrieve_paper(
    driver: Driver,
    entity_value: str,
) -> dict[str, Any]:
    if entity_value.startswith(
        "https://openalex.org/W"
    ):
        profile = get_paper_knowledge(
            driver,
            entity_value,
        )

        if profile is None:
            return {
                "entity_type": "paper",
                "entity_value": entity_value,
                "status": "not_found",
                "results": [],
            }

        return {
            "entity_type": "paper",
            "entity_value": entity_value,
            "status": "ok",
            "results": [
                profile
            ],
        }

    candidates = find_papers_by_title(
        driver,
        entity_value,
    )

    if not candidates:
        return {
            "entity_type": "paper",
            "entity_value": entity_value,
            "status": "not_found",
            "results": [],
        }

    best_rank = candidates[0][
        "match_rank"
    ]

    best_candidates = [
        candidate
        for candidate in candidates
        if candidate["match_rank"]
        == best_rank
    ]

    if len(best_candidates) > 1:
        return {
            "entity_type": "paper",
            "entity_value": entity_value,
            "status": "ambiguous",
            "results": best_candidates,
        }

    paper_id = best_candidates[0][
        "openalex_id"
    ]

    profile = get_paper_knowledge(
        driver,
        paper_id,
    )

    if profile is None:
        return {
            "entity_type": "paper",
            "entity_value": entity_value,
            "status": "not_found",
            "results": [],
        }

    return {
        "entity_type": "paper",
        "entity_value": entity_value,
        "status": "ok",
        "results": [
            profile
        ],
    }