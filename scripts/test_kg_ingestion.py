from config.settings import (
    NEO4J_DATABASE,
)
from knowledge_graph.client import (
    create_neo4j_driver,
)
from knowledge_graph.ingestion import (
    ingest_papers,
)
from tools.paper_search import (
    search_papers,
)


def get_graph_stats(
    driver,
    paper_ids: list[str],
) -> dict[str, int]:
    records, _, _ = driver.execute_query(
        """
        MATCH (p:Paper)
        WHERE p.openalex_id IN $paper_ids

        OPTIONAL MATCH (
            a:Author
        )-[
            authored:AUTHORED
        ]->(
            p
        )

        OPTIONAL MATCH (
            p
        )-[
            published:PUBLISHED_IN
        ]->(
            v:Venue
        )

        RETURN
            count(
                DISTINCT p
            ) AS papers,

            count(
                DISTINCT a
            ) AS authors,

            count(
                DISTINCT v
            ) AS venues,

            count(
                DISTINCT authored
            ) AS authored_relationships,

            count(
                DISTINCT published
            ) AS published_relationships
        """,
        paper_ids=paper_ids,
        database_=NEO4J_DATABASE,
    )

    record = records[0]

    return {
        "papers": record[
            "papers"
        ],
        "authors": record[
            "authors"
        ],
        "venues": record[
            "venues"
        ],
        "authored_relationships": record[
            "authored_relationships"
        ],
        "published_relationships": record[
            "published_relationships"
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
            "empathetic response generation"
        ),
        limit=3,
        require_abstract=True,
    )

    if not papers:
        raise RuntimeError(
            "OpenAlex returned no papers."
        )

    paper_ids = [
        paper["openalex_id"]
        for paper in papers
        if paper.get("openalex_id")
    ]

    print(
        f"Retrieved papers: "
        f"{len(papers)}"
    )

    with create_neo4j_driver() as driver:
        driver.verify_connectivity()

        print()
        print("First ingestion...")

        ingest_papers(
            driver,
            papers,
        )

        first_stats = get_graph_stats(
            driver,
            paper_ids,
        )

        print_stats(
            "AFTER FIRST INGESTION",
            first_stats,
        )

        print()
        print("Second ingestion...")

        ingest_papers(
            driver,
            papers,
        )

        second_stats = get_graph_stats(
            driver,
            paper_ids,
        )

        print_stats(
            "AFTER SECOND INGESTION",
            second_stats,
        )

        if first_stats != second_stats:
            raise AssertionError(
                "Knowledge graph ingestion "
                "is not idempotent."
            )

    print()
    print("=" * 70)
    print(
        "KNOWLEDGE GRAPH INGESTION: PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()