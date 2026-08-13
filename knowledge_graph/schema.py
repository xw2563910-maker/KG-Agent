from neo4j import Driver

from config.settings import (
    NEO4J_DATABASE,
)


SCHEMA_QUERIES = [
    """
    CREATE CONSTRAINT paper_openalex_id_unique
    IF NOT EXISTS
    FOR (p:Paper)
    REQUIRE p.openalex_id IS UNIQUE
    """,

    """
    CREATE CONSTRAINT author_openalex_id_unique
    IF NOT EXISTS
    FOR (a:Author)
    REQUIRE a.openalex_id IS UNIQUE
    """,

    """
    CREATE CONSTRAINT venue_openalex_id_unique
    IF NOT EXISTS
    FOR (v:Venue)
    REQUIRE v.openalex_id IS UNIQUE
    """,

    """
    CREATE CONSTRAINT method_normalized_name_unique
    IF NOT EXISTS
    FOR (m:Method)
    REQUIRE m.normalized_name IS UNIQUE
    """,

    """
    CREATE CONSTRAINT dataset_normalized_name_unique
    IF NOT EXISTS
    FOR (d:Dataset)
    REQUIRE d.normalized_name IS UNIQUE
    """,

    """
    CREATE CONSTRAINT task_normalized_name_unique
    IF NOT EXISTS
    FOR (t:Task)
    REQUIRE t.normalized_name IS UNIQUE
    """,
]


def initialize_schema(
    driver: Driver,
) -> None:
    for query in SCHEMA_QUERIES:
        driver.execute_query(
            query,
            database_=NEO4J_DATABASE,
        )