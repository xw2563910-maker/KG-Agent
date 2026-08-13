from agent.graph import (
    agent_graph,
)
from config.settings import (
    NEO4J_DATABASE,
)
from knowledge_graph.client import (
    create_neo4j_driver,
)


def verify_selected_papers_in_kg(
    papers: list[dict],
) -> None:
    paper_ids = [
        paper.get("openalex_id")
        for paper in papers
        if paper.get("openalex_id")
    ]

    if not paper_ids:
        raise AssertionError(
            "Selected papers have "
            "no OpenAlex IDs."
        )

    with create_neo4j_driver() as driver:
        driver.verify_connectivity()

        records, _, _ = (
            driver.execute_query(
                """
                UNWIND $paper_ids AS paper_id

                OPTIONAL MATCH (
                    p:Paper {
                        openalex_id: paper_id
                    }
                )

                RETURN
                    paper_id,
                    count(p) AS paper_count
                """,
                paper_ids=paper_ids,
                database_=NEO4J_DATABASE,
            )
        )

    missing = []

    for record in records:
        paper_id = record[
            "paper_id"
        ]

        paper_count = record[
            "paper_count"
        ]

        print(
            f"KG Paper: "
            f"{paper_id} "
            f"-> count={paper_count}"
        )

        if paper_count != 1:
            missing.append(
                paper_id
            )

    if missing:
        raise AssertionError(
            "Some selected papers were "
            "not correctly ingested: "
            f"{missing}"
        )


def main():
    question = (
        "请分析共情回复生成的研究趋势"
    )

    print("=" * 70)
    print("QUESTION")
    print("=" * 70)
    print(
        question
    )

    result = agent_graph.invoke(
        {
            "question": question
        }
    )

    print()
    print("=" * 70)
    print("LANGGRAPH RESULT")
    print("=" * 70)

    route = result.get(
        "route"
    )

    print(
        f"Route: {route}"
    )

    if route != "research":
        raise AssertionError(
            "Expected research route, "
            f"got {route!r}."
        )

    papers = result.get(
        "papers",
        [],
    )

    if not papers:
        raise AssertionError(
            "Research route returned "
            "no selected papers."
        )

    print(
        f"Selected papers: "
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

    report = result.get(
        "kg_ingestion_report"
    )

    if not report:
        raise AssertionError(
            "KG ingestion report "
            "is missing."
        )

    print()
    print("=" * 70)
    print("KG AUTO-INGESTION REPORT")
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
            "- "
            f"{error.get('stage')} | "
            f"{error.get('title')} | "
            f"{error.get('error')}"
        )

    if report[
        "total"
    ] != len(papers):
        raise AssertionError(
            "KG ingestion report total "
            "does not match selected "
            "paper count."
        )

    if report[
        "metadata_ingested"
    ] != len(papers):
        raise AssertionError(
            "Not all selected paper "
            "metadata was ingested."
        )

    print()
    print("=" * 70)
    print("VERIFY NEO4J")
    print("=" * 70)

    verify_selected_papers_in_kg(
        papers
    )

    answer = result.get(
        "answer"
    )

    if not answer:
        raise AssertionError(
            "Research answer is missing."
        )

    print()
    print("=" * 70)
    print("RESEARCH ANSWER")
    print("=" * 70)
    print(
        answer
    )

    print()
    print("=" * 70)
    print(
        "LANGGRAPH RESEARCH -> KG "
        "AUTO-INGESTION: PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()