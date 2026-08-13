from config.settings import (
    NEO4J_DATABASE,
)
from knowledge_graph.client import (
    create_neo4j_driver,
)
from knowledge_graph.entity_extraction import (
    extract_scientific_entities,
)
from knowledge_graph.ingestion import (
    ingest_paper,
)
from knowledge_graph.semantic_ingestion import (
    ingest_scientific_entities,
)
from tools.paper_search import (
    search_papers,
)


def get_semantic_stats(
    driver,
    paper_id: str,
) -> dict[str, int]:
    records, _, _ = driver.execute_query(
        """
        MATCH (
            p:Paper {
                openalex_id: $paper_id
            }
        )

        OPTIONAL MATCH (
            p
        )-[
            method_rel:USES_METHOD
        ]->(
            m:Method
        )

        OPTIONAL MATCH (
            p
        )-[
            dataset_rel:EVALUATED_ON
        ]->(
            d:Dataset
        )

        OPTIONAL MATCH (
            p
        )-[
            task_rel:ADDRESSES_TASK
        ]->(
            t:Task
        )

        RETURN
            count(
                DISTINCT m
            ) AS methods,

            count(
                DISTINCT d
            ) AS datasets,

            count(
                DISTINCT t
            ) AS tasks,

            count(
                DISTINCT method_rel
            ) AS method_relationships,

            count(
                DISTINCT dataset_rel
            ) AS dataset_relationships,

            count(
                DISTINCT task_rel
            ) AS task_relationships
        """,
        paper_id=paper_id,
        database_=NEO4J_DATABASE,
    )

    record = records[0]

    return {
        "methods": record["methods"],
        "datasets": record["datasets"],
        "tasks": record["tasks"],
        "method_relationships": record[
            "method_relationships"
        ],
        "dataset_relationships": record[
            "dataset_relationships"
        ],
        "task_relationships": record[
            "task_relationships"
        ],
    }


def print_stats(
    title: str,
    stats: dict[str, int],
) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    for key, value in stats.items():
        print(
            f"{key}: {value}"
        )


def main():
    papers = search_papers(
        query=(
            "CEM Commonsense-Aware "
            "Empathetic Response Generation"
        ),
        limit=1,
        require_abstract=True,
    )

    if not papers:
        raise RuntimeError(
            "OpenAlex returned no papers."
        )

    paper = papers[0]

    paper_id = paper.get(
        "openalex_id"
    )

    if not paper_id:
        raise RuntimeError(
            "Paper OpenAlex ID is missing."
        )

    print("=" * 70)
    print("PAPER")
    print("=" * 70)

    print(
        f"Title: {paper.get('title')}"
    )

    print(
        f"OpenAlex ID: {paper_id}"
    )

    print()

    print("=" * 70)
    print("ENTITY EXTRACTION")
    print("=" * 70)

    entities = extract_scientific_entities(
        paper
    )

    print(
        f"Methods: "
        f"{entities['methods']}"
    )

    print(
        f"Datasets: "
        f"{entities['datasets']}"
    )

    print(
        f"Tasks: "
        f"{entities['tasks']}"
    )

    with create_neo4j_driver() as driver:
        driver.verify_connectivity()

        # Make sure the Paper node exists.
        ingest_paper(
            driver,
            paper,
        )

        print()
        print(
            "First semantic ingestion..."
        )

        ingest_scientific_entities(
            driver,
            paper_id,
            entities,
        )

        first_stats = get_semantic_stats(
            driver,
            paper_id,
        )

        print_stats(
            "AFTER FIRST SEMANTIC INGESTION",
            first_stats,
        )

        print()
        print(
            "Second semantic ingestion..."
        )

        ingest_scientific_entities(
            driver,
            paper_id,
            entities,
        )

        second_stats = get_semantic_stats(
            driver,
            paper_id,
        )

        print_stats(
            "AFTER SECOND SEMANTIC INGESTION",
            second_stats,
        )

        if first_stats != second_stats:
            raise AssertionError(
                "Semantic ingestion "
                "is not idempotent."
            )

    print()
    print("=" * 70)
    print(
        "SEMANTIC KNOWLEDGE INGESTION: PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()