from agent.kg_answer import (
    answer_kg_question,
)
from agent.kg_retrieval import (
    retrieve_kg,
)
from knowledge_graph.client import (
    create_neo4j_driver,
)


def main():
    with create_neo4j_driver() as driver:
        driver.verify_connectivity()

        print("=" * 70)
        print("TEST 1: DATASET QUESTION")
        print("=" * 70)

        dataset_result = retrieve_kg(
            driver,
            {
                "entity_type": "dataset",
                "entity_value": (
                    "EmpatheticDialogues"
                ),
            },
        )

        dataset_answer = answer_kg_question(
            question=(
                "知识图谱中哪些论文使用了 "
                "EmpatheticDialogues 数据集？"
            ),
            retrieval_result=(
                dataset_result
            ),
        )

        print(
            dataset_answer
        )

        if not dataset_answer.strip():
            raise AssertionError(
                "Dataset answer is empty."
            )

        print()
        print("=" * 70)
        print("TEST 2: PAPER QUESTION")
        print("=" * 70)

        paper_result = retrieve_kg(
            driver,
            {
                "entity_type": "paper",
                "entity_value": "CEM",
            },
        )

        paper_answer = answer_kg_question(
            question=(
                "请介绍知识图谱中的 "
                "CEM 论文。"
            ),
            retrieval_result=(
                paper_result
            ),
        )

        print(
            paper_answer
        )

        if not paper_answer.strip():
            raise AssertionError(
                "Paper answer is empty."
            )

        print()
        print("=" * 70)
        print("TEST 3: NOT FOUND")
        print("=" * 70)

        not_found_result = {
            "entity_type": "dataset",
            "entity_value": (
                "DefinitelyMissingDataset"
            ),
            "status": "not_found",
            "results": [],
        }

        not_found_answer = (
            answer_kg_question(
                question=(
                    "哪些论文使用了 "
                    "DefinitelyMissingDataset？"
                ),
                retrieval_result=(
                    not_found_result
                ),
            )
        )

        print(
            not_found_answer
        )

        if (
            "没有找到"
            not in not_found_answer
        ):
            raise AssertionError(
                "Not-found answer does not "
                "report missing evidence."
            )

    print()
    print("=" * 70)
    print(
        "KG ANSWER LAYER: PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()