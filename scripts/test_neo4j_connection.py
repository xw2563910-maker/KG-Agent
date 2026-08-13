from knowledge_graph.client import (
    create_neo4j_driver,
)
from config.settings import (
    NEO4J_DATABASE,
)


def main():
    with create_neo4j_driver() as driver:
        driver.verify_connectivity()

        print(
            "Neo4j connectivity: OK"
        )

        records, _, _ = driver.execute_query(
            """
            MATCH
                (a:Author)-[:AUTHORED]->(p:Paper)
            RETURN
                a.name AS author,
                p.title AS paper
            """,
            database_=NEO4J_DATABASE,
        )

        print()
        print("Knowledge graph records:")

        for record in records:
            print(
                f"{record['author']} "
                f"--AUTHORED--> "
                f"{record['paper']}"
            )


if __name__ == "__main__":
    main()