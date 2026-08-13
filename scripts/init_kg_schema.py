from knowledge_graph.client import (
    create_neo4j_driver,
)
from knowledge_graph.schema import (
    initialize_schema,
)


def main():
    with create_neo4j_driver() as driver:
        driver.verify_connectivity()

        initialize_schema(
            driver
        )

    print(
        "Knowledge graph schema initialized."
    )


if __name__ == "__main__":
    main()