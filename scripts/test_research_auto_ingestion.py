from config.settings import (
    NEO4J_DATABASE,
)
from knowledge_graph.auto_ingestion import (
    ingest_research_papers,
)
from knowledge_graph.client import (
    create_neo4j_driver,
)
from tools.paper_search import (
    search_papers,
)


def verify_paper_uniqueness(
    driver,
    paper_ids: list[str],
) -> None:
    for paper_id in paper_ids:
        records, _, _ = (
            driver.execute_query(
                """
                MATCH (
                    p:Paper {
                        openalex_id: $paper_id
                    }
                )

                RETURN count(p) AS count
                """,
                paper_id=paper_id,
                database_=NEO4J_DATABASE,
            )
        )

        count = records[0][
            "count"
        ]

        if count != 1:
            raise AssertionError(
                "Expected exactly one "
                "Paper node for "
                f"{paper_id}, "
                f"got {count}."
            )


def print_report(
    title: str,
    report: dict,
) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    print(
        f"total: "
        f"{report['total']}"
    )

    print(
        f"metadata_ingested: "
        f"{report['metadata_ingested']}"
    )

    print(
        f"semantic_ingested: "
        f"{report['semantic_ingested']}"
    )

    print(
        f"skipped: "
        f"{report['skipped']}"
    )

    print(
        f"errors: "
        f"{len(report['errors'])}"
    )

    for error in report[
        "errors"
    ]:
        print(
            f"- "
            f"{error['stage']} | "
            f"{error['title']} | "
            f"{error['error']}"
        )


def main():
    print("=" * 70)
    print(
        "SEARCH REAL PAPERS"
    )
    print("=" * 70)

    papers = search_papers(
        query=(
            "empathetic response generation"
        ),
        limit=2,
        require_abstract=True,
    )

    if not papers:
        raise RuntimeError(
            "OpenAlex returned "
            "no papers."
        )

    print(
        f"Retrieved papers: "
        f"{len(papers)}"
    )

    for index, paper in enumerate(
        papers,
        start=1,
    ):
        print(
            f"{index}. "
            f"{paper.get('title')}"
        )

    paper_ids = [
        paper["openalex_id"]
        for paper in papers
        if paper.get(
            "openalex_id"
        )
    ]

    with create_neo4j_driver() as driver:
        driver.verify_connectivity()

        # ========================================================
        # First ingestion
        # ========================================================

        first_report = (
            ingest_research_papers(
                driver,
                papers,
            )
        )

        print_report(
            "AFTER FIRST AUTO INGESTION",
            first_report,
        )

        verify_paper_uniqueness(
            driver,
            paper_ids,
        )

        # ========================================================
        # Second ingestion
        # ========================================================

        second_report = (
            ingest_research_papers(
                driver,
                papers,
            )
        )

        print_report(
            "AFTER SECOND AUTO INGESTION",
            second_report,
        )

        verify_paper_uniqueness(
            driver,
            paper_ids,
        )

    if (
        first_report[
            "metadata_ingested"
        ]
        != len(papers)
    ):
        raise AssertionError(
            "Not all paper metadata "
            "was ingested."
        )

    if (
        second_report[
            "metadata_ingested"
        ]
        != len(papers)
    ):
        raise AssertionError(
            "Second metadata ingestion "
            "did not process all papers."
        )

    print()
    print("=" * 70)
    print(
        "RESEARCH AUTO INGESTION: PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()