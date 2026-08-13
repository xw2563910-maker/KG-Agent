from neo4j import Driver, GraphDatabase

from config.settings import (
    NEO4J_DATABASE,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USERNAME,
    validate_neo4j_config,
)


def create_neo4j_driver() -> Driver:
    validate_neo4j_config()

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(
            NEO4J_USERNAME,
            NEO4J_PASSWORD,
        ),
    )

    return driver


def verify_neo4j_connection() -> None:
    with create_neo4j_driver() as driver:
        driver.verify_connectivity()

        driver.execute_query(
            "RETURN 1 AS ok",
            database_=NEO4J_DATABASE,
        )