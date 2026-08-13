from agent.kg_planner import (
    build_kg_query_plan,
)


def test_question(
    question: str,
    expected_type: str,
) -> None:
    print("=" * 70)

    print(
        f"Question: {question}"
    )

    plan = build_kg_query_plan(
        question
    )

    print(
        f"Entity type: "
        f"{plan['entity_type']}"
    )

    print(
        f"Entity value: "
        f"{plan['entity_value']}"
    )

    if (
        plan["entity_type"]
        != expected_type
    ):
        raise AssertionError(
            f"Expected entity type "
            f"{expected_type!r}, "
            f"got "
            f"{plan['entity_type']!r}"
        )

    print(
        "Result: PASSED"
    )


def main():
    test_question(
        question=(
            "知识图谱中哪些论文使用了 "
            "CEM 方法？"
        ),
        expected_type="method",
    )

    test_question(
        question=(
            "知识图谱中哪些论文使用了 "
            "EmpatheticDialogues 数据集？"
        ),
        expected_type="dataset",
    )

    test_question(
        question=(
            "知识图谱中哪些论文研究 "
            "empathetic response generation？"
        ),
        expected_type="task",
    )

    test_question(
        question=(
            "请查看知识图谱中 "
            "CEM 这篇论文的信息。"
        ),
        expected_type="paper",
    )

    print()
    print("=" * 70)
    print(
        "KG QUERY PLANNER: PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()