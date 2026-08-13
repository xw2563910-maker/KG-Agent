from agent.graph import (
    agent_graph,
)


def run_kg_test(
    question: str,
    expected_entity_type: str,
) -> None:
    print("=" * 70)

    print(
        f"Question: {question}"
    )

    result = agent_graph.invoke(
        {
            "question": question
        }
    )

    # ============================================================
    # Validate Route
    # ============================================================

    route = result.get(
        "route"
    )

    print(
        f"Route: {route}"
    )

    if route != "kg":
        raise AssertionError(
            "Expected LangGraph route "
            f"'kg', got {route!r}."
        )

    # ============================================================
    # Validate KG Query Plan
    # ============================================================

    plan = result.get(
        "kg_query_plan"
    )

    if not plan:
        raise AssertionError(
            "KG query plan is missing."
        )

    entity_type = plan.get(
        "entity_type"
    )

    entity_value = plan.get(
        "entity_value"
    )

    print(
        f"Entity type: "
        f"{entity_type}"
    )

    print(
        f"Entity value: "
        f"{entity_value}"
    )

    if entity_type != expected_entity_type:
        raise AssertionError(
            "Unexpected KG entity type. "
            f"Expected "
            f"{expected_entity_type!r}, "
            f"got {entity_type!r}."
        )

    # ============================================================
    # Validate KG Retrieval
    # ============================================================

    retrieval = result.get(
        "kg_retrieval_result"
    )

    if not retrieval:
        raise AssertionError(
            "KG retrieval result "
            "is missing."
        )

    retrieval_status = retrieval.get(
        "status"
    )

    retrieval_results = retrieval.get(
        "results",
        [],
    )

    print(
        f"Retrieval status: "
        f"{retrieval_status}"
    )

    print(
        f"Retrieval results: "
        f"{len(retrieval_results)}"
    )

    if retrieval_status != "ok":
        raise AssertionError(
            "Expected successful KG "
            f"retrieval, got "
            f"{retrieval_status!r}."
        )

    if not retrieval_results:
        raise AssertionError(
            "KG retrieval returned "
            "no results."
        )

    # ============================================================
    # Validate Final Answer
    # ============================================================

    answer = result.get(
        "answer"
    )

    if not answer:
        raise AssertionError(
            "KG answer is missing."
        )

    print()
    print("Answer:")
    print(
        answer
    )

    print()
    print(
        "Result: PASSED"
    )


def main():
    # ============================================================
    # Test 1: Dataset Query
    # ============================================================

    run_kg_test(
        question=(
            "知识图谱中哪些论文使用了 "
            "EmpatheticDialogues 数据集？"
        ),
        expected_entity_type="dataset",
    )

    # ============================================================
    # Test 2: Paper Query
    # ============================================================

    run_kg_test(
        question=(
            "请介绍知识图谱中的 "
            "CEM 论文。"
        ),
        expected_entity_type="paper",
    )

    # ============================================================
    # Final Validation
    # ============================================================

    print()
    print("=" * 70)
    print(
        "LANGGRAPH KG ROUTE: PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()