from agent.kg_retrieval import (
    retrieve_kg,
)
from knowledge_graph.client import (
    create_neo4j_driver,
)


CEM_PAPER_ID = (
    "https://openalex.org/W3200833038"
)


def run_test(
    driver,
    plan: dict,
) -> dict:
    print("=" * 70)

    print(
        f"Plan: {plan}"
    )

    result = retrieve_kg(
        driver,
        plan,
    )

    print(
        f"Status: {result['status']}"
    )

    print(
        f"Results: {len(result['results'])}"
    )

    for item in result["results"]:
        print(
            f"  - {item.get('title')}"
        )

    return result


def assert_contains_cem(
    result: dict,
) -> None:
    if result["status"] != "ok":
        raise AssertionError(
            "Expected KG retrieval "
            "status to be 'ok'."
        )

    paper_ids = {
        item.get("openalex_id")
        for item in result["results"]
    }

    if CEM_PAPER_ID not in paper_ids:
        raise AssertionError(
            "CEM paper was not found "
            "in KG retrieval results."
        )


def main():
    with create_neo4j_driver() as driver:
        driver.verify_connectivity()

        method_result = run_test(
            driver,
            {
                "entity_type": "method",
                "entity_value": "CEM",
            },
        )

        assert_contains_cem(
            method_result
        )

        dataset_result = run_test(
            driver,
            {
                "entity_type": "dataset",
                "entity_value": (
                    "EmpatheticDialogues"
                ),
            },
        )

        assert_contains_cem(
            dataset_result
        )

        task_result = run_test(
            driver,
            {
                "entity_type": "task",
                "entity_value": (
                    "Empathetic "
                    "Response Generation"
                ),
            },
        )

        assert_contains_cem(
            task_result
        )

        paper_result = run_test(
            driver,
            {
                "entity_type": "paper",
                "entity_value": "CEM",
            },
        )

        assert_contains_cem(
            paper_result
        )

        profile = (
            paper_result["results"][0]
        )

        if "CEM" not in profile[
            "methods"
        ]:
            raise AssertionError(
                "CEM method was not found "
                "in the Paper profile."
            )

        if "EMPATHETICDIALOGUES" not in (
            profile["datasets"]
        ):
            raise AssertionError(
                "Expected dataset was not "
                "found in Paper profile."
            )

    print()
    print("=" * 70)
    print(
        "KG RETRIEVAL: PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()