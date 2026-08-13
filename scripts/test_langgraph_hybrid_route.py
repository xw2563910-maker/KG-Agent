import sys
from pathlib import Path

from agent.graph import agent_graph


def main():
    if len(sys.argv) != 2:
        raise ValueError(
            "Usage: python -m "
            "scripts.test_langgraph_hybrid_route "
            '"<pdf_path>"'
        )

    pdf_path = Path(sys.argv[1])

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF does not exist: {pdf_path}"
        )

    question = (
        "结合这篇 PDF 和知识图谱中的 CEM 论文信息，"
        "请区分两类证据分别支持了哪些结论，并说明"
        "它们是否一致。"
    )

    print("=" * 70)
    print("QUESTION")
    print("=" * 70)
    print(question)
    print(f"PDF: {pdf_path}")

    result = agent_graph.invoke(
        {
            "question": question,
            "pdf_path": str(pdf_path),
        }
    )

    route = result.get("route")
    plan = result.get("kg_query_plan")
    kg_result = result.get("kg_retrieval_result")
    chunks = result.get("retrieved_chunks", [])
    answer = result.get("answer")

    print()
    print("=" * 70)
    print("HYBRID STATE")
    print("=" * 70)
    print(f"Route: {route}")
    print(f"KG plan: {plan}")
    print(
        "KG status: "
        f"{kg_result.get('status') if kg_result else None}"
    )
    print(f"PDF chunks: {len(chunks)}")

    if route != "hybrid":
        raise AssertionError(
            f"Expected hybrid route, got {route!r}."
        )

    if not plan:
        raise AssertionError(
            "Hybrid route produced no KG query plan."
        )

    if not kg_result:
        raise AssertionError(
            "Hybrid route produced no KG retrieval result."
        )

    if not chunks:
        raise AssertionError(
            "Hybrid route produced no PDF evidence."
        )

    if not answer:
        raise AssertionError(
            "Hybrid route produced no answer."
        )

    print()
    print("=" * 70)
    print("HYBRID ANSWER")
    print("=" * 70)
    print(answer)

    print()
    print("=" * 70)
    print("LANGGRAPH HYBRID ROUTE: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
