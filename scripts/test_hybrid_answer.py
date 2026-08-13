from agent.hybrid_answer import (
    answer_hybrid_question,
)
from agent.kg_retrieval import (
    retrieve_kg,
)
from knowledge_graph.client import (
    create_neo4j_driver,
)


def main():
    question = (
        "结合知识图谱和 PDF 证据，"
        "请说明 CEM 论文研究了什么任务、"
        "使用了什么数据集，以及 PDF 证据"
        "还说明了什么方法信息？"
    )

    print("=" * 70)
    print("QUESTION")
    print("=" * 70)
    print(
        question
    )

    # ============================================================
    # Real Neo4j Evidence
    # ============================================================

    with create_neo4j_driver() as driver:
        driver.verify_connectivity()

        kg_result = retrieve_kg(
            driver,
            {
                "entity_type": "paper",
                "entity_value": "CEM",
            },
        )

    print()
    print("=" * 70)
    print("KG EVIDENCE")
    print("=" * 70)

    print(
        f"status: "
        f"{kg_result['status']}"
    )

    print(
        f"results: "
        f"{len(kg_result['results'])}"
    )

    if kg_result["status"] != "ok":
        raise AssertionError(
            "Expected successful "
            "CEM KG retrieval."
        )

    # ============================================================
    # Simulated output of PDF RAG retrieval
    #
    # Phase 8 already tested PDF loading, chunking,
    # embedding, FAISS retrieval separately.
    # ============================================================

    retrieved_chunks = [
        {
            "page": 1,
            "score": 0.91,
            "text": (
                "CEM is a commonsense-aware empathetic "
                "response generation framework. The model "
                "incorporates commonsense knowledge to "
                "better understand the user's situation "
                "and generate empathetic responses."
            ),
        }
    ]

    print()
    print("=" * 70)
    print("PDF RAG EVIDENCE")
    print("=" * 70)

    for chunk in retrieved_chunks:
        print(
            f"Page: {chunk.get('page')}"
        )

        print(
            f"Score: {chunk.get('score')}"
        )

        print(
            f"Text: {chunk.get('text')}"
        )

    # ============================================================
    # Hybrid Fusion
    # ============================================================

    answer = answer_hybrid_question(
        question=question,
        kg_retrieval_result=kg_result,
        retrieved_chunks=retrieved_chunks,
    )

    print()
    print("=" * 70)
    print("HYBRID ANSWER")
    print("=" * 70)

    print(
        answer
    )

    if not answer:
        raise AssertionError(
            "Hybrid answer is empty."
        )

    print()
    print("=" * 70)
    print(
        "HYBRID ANSWER LAYER: PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()