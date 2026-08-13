from knowledge_graph.client import (
    create_neo4j_driver,
)
from knowledge_graph.query import (
    find_papers_by_dataset,
    find_papers_by_method,
    find_papers_by_task,
    get_paper_knowledge,
)


CEM_PAPER_ID = (
    "https://openalex.org/W3200833038"
)


def main():
    with create_neo4j_driver() as driver:
        driver.verify_connectivity()

        print("=" * 70)
        print("PAPER KNOWLEDGE PROFILE")
        print("=" * 70)

        profile = get_paper_knowledge(
            driver,
            CEM_PAPER_ID,
        )

        if profile is None:
            raise AssertionError(
                "CEM Paper was not found "
                "in Neo4j."
            )

        print(
            f"Title: {profile['title']}"
        )

        print(
            f"Year: {profile['year']}"
        )

        print(
            f"Authors: {profile['authors']}"
        )

        print(
            f"Venue: {profile['venue']}"
        )

        print(
            f"Methods: {profile['methods']}"
        )

        print(
            f"Datasets: {profile['datasets']}"
        )

        print(
            f"Tasks: {profile['tasks']}"
        )

        print()
        print("=" * 70)
        print("QUERY BY METHOD")
        print("=" * 70)

        method_papers = find_papers_by_method(
            driver,
            " CEM ",
        )

        for paper in method_papers:
            print(
                f"- {paper['title']}"
            )

        print()
        print("=" * 70)
        print("QUERY BY DATASET")
        print("=" * 70)

        dataset_papers = (
            find_papers_by_dataset(
                driver,
                "EmpatheticDialogues",
            )
        )

        for paper in dataset_papers:
            print(
                f"- {paper['title']}"
            )

        print()
        print("=" * 70)
        print("QUERY BY TASK")
        print("=" * 70)

        task_papers = find_papers_by_task(
            driver,
            (
                "  Empathetic   "
                "Response Generation  "
            ),
        )

        for paper in task_papers:
            print(
                f"- {paper['title']}"
            )

        if CEM_PAPER_ID not in {
            paper["openalex_id"]
            for paper in method_papers
        }:
            raise AssertionError(
                "CEM was not found "
                "by method."
            )

        if CEM_PAPER_ID not in {
            paper["openalex_id"]
            for paper in dataset_papers
        }:
            raise AssertionError(
                "CEM was not found "
                "by dataset."
            )

        if CEM_PAPER_ID not in {
            paper["openalex_id"]
            for paper in task_papers
        }:
            raise AssertionError(
                "CEM was not found "
                "by task."
            )

    print()
    print("=" * 70)
    print(
        "KNOWLEDGE GRAPH QUERIES: PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()