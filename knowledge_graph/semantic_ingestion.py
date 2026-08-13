from typing import Any

from neo4j import Driver

from config.settings import NEO4J_DATABASE
from knowledge_graph.normalization import (
    normalize_entity_name,
)


METHOD_QUERY = """
MATCH (
    p:Paper {
        openalex_id: $paper_id
    }
)

UNWIND $entities AS entity

MERGE (
    m:Method {
        normalized_name: entity.normalized_name
    }
)

ON CREATE SET
    m.name = entity.name

MERGE (
    p
)-[
    r:USES_METHOD
]->(
    m
)

ON CREATE SET
    r.source = "llm_extraction",
    r.evidence_scope = "title_abstract"
"""


DATASET_QUERY = """
MATCH (
    p:Paper {
        openalex_id: $paper_id
    }
)

UNWIND $entities AS entity

MERGE (
    d:Dataset {
        normalized_name: entity.normalized_name
    }
)

ON CREATE SET
    d.name = entity.name

MERGE (
    p
)-[
    r:EVALUATED_ON
]->(
    d
)

ON CREATE SET
    r.source = "llm_extraction",
    r.evidence_scope = "title_abstract"
"""


TASK_QUERY = """
MATCH (
    p:Paper {
        openalex_id: $paper_id
    }
)

UNWIND $entities AS entity

MERGE (
    t:Task {
        normalized_name: entity.normalized_name
    }
)

ON CREATE SET
    t.name = entity.name

MERGE (
    p
)-[
    r:ADDRESSES_TASK
]->(
    t
)

ON CREATE SET
    r.source = "llm_extraction",
    r.evidence_scope = "title_abstract"
"""


def _prepare_entities(
    names: Any,
) -> list[dict[str, str]]:
    if not isinstance(names, list):
        return []

    entities = []
    seen = set()

    for name in names:
        if not isinstance(name, str):
            continue

        display_name = name.strip()

        if not display_name:
            continue

        normalized_name = (
            normalize_entity_name(
                display_name
            )
        )

        if normalized_name in seen:
            continue

        seen.add(
            normalized_name
        )

        entities.append(
            {
                "name": display_name,
                "normalized_name": (
                    normalized_name
                ),
            }
        )

    return entities


def _ensure_paper_exists(
    driver: Driver,
    paper_id: str,
) -> None:
    records, _, _ = driver.execute_query(
        """
        MATCH (
            p:Paper {
                openalex_id: $paper_id
            }
        )

        RETURN count(p) AS paper_count
        """,
        paper_id=paper_id,
        database_=NEO4J_DATABASE,
    )

    if not records:
        raise RuntimeError(
            "Failed to verify Paper node."
        )

    if records[0]["paper_count"] != 1:
        raise ValueError(
            "Paper must exist in Neo4j "
            "before semantic ingestion."
        )


def ingest_scientific_entities(
    driver: Driver,
    paper_id: str,
    entities: dict[str, Any],
) -> None:
    paper_id = str(
        paper_id or ""
    ).strip()

    if not paper_id:
        raise ValueError(
            "Paper OpenAlex ID is required."
        )

    if not isinstance(entities, dict):
        raise TypeError(
            "Entities must be a dictionary."
        )

    _ensure_paper_exists(
        driver,
        paper_id,
    )

    methods = _prepare_entities(
        entities.get("methods")
    )

    datasets = _prepare_entities(
        entities.get("datasets")
    )

    tasks = _prepare_entities(
        entities.get("tasks")
    )

    if methods:
        driver.execute_query(
            METHOD_QUERY,
            paper_id=paper_id,
            entities=methods,
            database_=NEO4J_DATABASE,
        )

    if datasets:
        driver.execute_query(
            DATASET_QUERY,
            paper_id=paper_id,
            entities=datasets,
            database_=NEO4J_DATABASE,
        )

    if tasks:
        driver.execute_query(
            TASK_QUERY,
            paper_id=paper_id,
            entities=tasks,
            database_=NEO4J_DATABASE,
        )