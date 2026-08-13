from typing import Any

from neo4j import Driver

from config.settings import NEO4J_DATABASE
from knowledge_graph.normalization import (
    normalize_entity_name,
)


def get_paper_knowledge(
    driver: Driver,
    paper_id: str,
) -> dict[str, Any] | None:
    paper_id = str(
        paper_id or ""
    ).strip()

    if not paper_id:
        raise ValueError(
            "Paper OpenAlex ID is required."
        )

    records, _, _ = driver.execute_query(
        """
        MATCH (
            p:Paper {
                openalex_id: $paper_id
            }
        )

        OPTIONAL MATCH (
            a:Author
        )-[
            :AUTHORED
        ]->(
            p
        )

        WITH
            p,
            collect(
                DISTINCT a.name
            ) AS authors

        OPTIONAL MATCH (
            p
        )-[
            :PUBLISHED_IN
        ]->(
            v:Venue
        )

        WITH
            p,
            authors,
            head(
                collect(
                    DISTINCT v.name
                )
            ) AS venue

        OPTIONAL MATCH (
            p
        )-[
            :USES_METHOD
        ]->(
            m:Method
        )

        WITH
            p,
            authors,
            venue,
            collect(
                DISTINCT m.name
            ) AS methods

        OPTIONAL MATCH (
            p
        )-[
            :EVALUATED_ON
        ]->(
            d:Dataset
        )

        WITH
            p,
            authors,
            venue,
            methods,
            collect(
                DISTINCT d.name
            ) AS datasets

        OPTIONAL MATCH (
            p
        )-[
            :ADDRESSES_TASK
        ]->(
            t:Task
        )

        RETURN
            p.openalex_id AS openalex_id,
            p.title AS title,
            p.year AS year,
            p.doi AS doi,
            authors,
            venue,
            methods,
            datasets,
            collect(
                DISTINCT t.name
            ) AS tasks
        """,
        paper_id=paper_id,
        database_=NEO4J_DATABASE,
    )

    if not records:
        return None

    return records[0].data()


def find_papers_by_method(
    driver: Driver,
    method_name: str,
) -> list[dict[str, Any]]:
    normalized_name = normalize_entity_name(
        method_name
    )

    records, _, _ = driver.execute_query(
        """
        MATCH (
            p:Paper
        )-[
            :USES_METHOD
        ]->(
            m:Method {
                normalized_name: $normalized_name
            }
        )

        RETURN DISTINCT
            p.openalex_id AS openalex_id,
            p.title AS title,
            p.year AS year,
            m.name AS method

        ORDER BY
            p.year DESC,
            p.title
        """,
        normalized_name=normalized_name,
        database_=NEO4J_DATABASE,
    )

    return [
        record.data()
        for record in records
    ]


def find_papers_by_dataset(
    driver: Driver,
    dataset_name: str,
) -> list[dict[str, Any]]:
    normalized_name = normalize_entity_name(
        dataset_name
    )

    records, _, _ = driver.execute_query(
        """
        MATCH (
            p:Paper
        )-[
            :EVALUATED_ON
        ]->(
            d:Dataset {
                normalized_name: $normalized_name
            }
        )

        RETURN DISTINCT
            p.openalex_id AS openalex_id,
            p.title AS title,
            p.year AS year,
            d.name AS dataset

        ORDER BY
            p.year DESC,
            p.title
        """,
        normalized_name=normalized_name,
        database_=NEO4J_DATABASE,
    )

    return [
        record.data()
        for record in records
    ]


def find_papers_by_task(
    driver: Driver,
    task_name: str,
) -> list[dict[str, Any]]:
    normalized_name = normalize_entity_name(
        task_name
    )

    records, _, _ = driver.execute_query(
        """
        MATCH (
            p:Paper
        )-[
            :ADDRESSES_TASK
        ]->(
            t:Task {
                normalized_name: $normalized_name
            }
        )

        RETURN DISTINCT
            p.openalex_id AS openalex_id,
            p.title AS title,
            p.year AS year,
            t.name AS task

        ORDER BY
            p.year DESC,
            p.title
        """,
        normalized_name=normalized_name,
        database_=NEO4J_DATABASE,
    )

    return [
        record.data()
        for record in records
    ]

def find_papers_by_title(
    driver: Driver,
    title_query: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    title_query = str(
        title_query or ""
    ).strip()

    if not title_query:
        raise ValueError(
            "Paper title query cannot be empty."
        )

    if not 1 <= limit <= 20:
        raise ValueError(
            "Paper title query limit must be "
            "between 1 and 20."
        )

    records, _, _ = driver.execute_query(
        """
        MATCH (p:Paper)

        WHERE
            p.title IS NOT NULL
            AND toLower(p.title)
                CONTAINS toLower($title_query)

        WITH
            p,
            CASE
                WHEN toLower(p.title)
                    = toLower($title_query)
                THEN 0

                WHEN toLower(p.title)
                    STARTS WITH toLower($title_query)
                THEN 1

                ELSE 2
            END AS match_rank

        RETURN
            p.openalex_id AS openalex_id,
            p.title AS title,
            p.year AS year,
            match_rank

        ORDER BY
            match_rank ASC,
            p.year DESC,
            p.title

        LIMIT $limit
        """,
        title_query=title_query,
        limit=limit,
        database_=NEO4J_DATABASE,
    )

    return [
        record.data()
        for record in records
    ]