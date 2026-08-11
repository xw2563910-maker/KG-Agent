import sys

from agent.graph import agent_graph


def test_general_route():
    print()
    print("=" * 70)
    print("TEST 1: GENERAL ROUTE")
    print("=" * 70)

    question = (
        "What is overfitting in machine learning?"
    )

    result = agent_graph.invoke(
        {
            "question": question
        }
    )

    route = result.get("route")
    answer = result.get("answer")

    if route != "general":
        raise AssertionError(
            f"Expected general route, got: {route}"
        )

    if not answer:
        raise AssertionError(
            "General route returned no answer."
        )

    print()
    print(f"Route: {route}")
    print()
    print("Answer:")
    print(answer)

    print()
    print("GENERAL ROUTE: PASSED")


def test_research_route():
    print()
    print("=" * 70)
    print("TEST 2: RESEARCH ROUTE")
    print("=" * 70)

    question = (
        "请分析近三年共情回复生成的研究趋势"
    )

    result = agent_graph.invoke(
        {
            "question": question
        }
    )

    route = result.get("route")

    candidate_papers = result.get(
        "candidate_papers",
        [],
    )

    quality_papers = result.get(
        "quality_papers",
        [],
    )

    papers = result.get(
        "papers",
        [],
    )

    answer = result.get("answer")

    if route != "research":
        raise AssertionError(
            f"Expected research route, got: {route}"
        )

    if not candidate_papers:
        raise AssertionError(
            "Research route retrieved "
            "no candidate papers."
        )

    if not quality_papers:
        raise AssertionError(
            "Research route produced "
            "no quality papers."
        )

    if not papers:
        raise AssertionError(
            "Research route selected "
            "no final papers."
        )

    if not answer:
        raise AssertionError(
            "Research route returned no answer."
        )

    print()
    print(f"Route: {route}")

    print(
        f"Candidate papers: "
        f"{len(candidate_papers)}"
    )

    print(
        f"Quality papers: "
        f"{len(quality_papers)}"
    )

    print(
        f"Selected papers: "
        f"{len(papers)}"
    )

    print()
    print("Answer:")
    print(answer)

    print()
    print("RESEARCH ROUTE: PASSED")


def test_pdf_route(
    pdf_path: str,
):
    print()
    print("=" * 70)
    print("TEST 3: PDF ROUTE")
    print("=" * 70)

    question = (
        "What reinforcement learning "
        "algorithm does EmpRL use?"
    )

    result = agent_graph.invoke(
        {
            "question": question,
            "pdf_path": pdf_path,
        }
    )

    route = result.get("route")

    retrieved_chunks = result.get(
        "retrieved_chunks",
        [],
    )

    answer = result.get("answer")

    if route != "pdf":
        raise AssertionError(
            f"Expected pdf route, got: {route}"
        )

    if not retrieved_chunks:
        raise AssertionError(
            "PDF route retrieved no chunks."
        )

    if not answer:
        raise AssertionError(
            "PDF route returned no answer."
        )

    print()
    print(f"Route: {route}")

    print(
        f"Retrieved chunks: "
        f"{len(retrieved_chunks)}"
    )

    print()
    print("Answer:")
    print(answer)

    print()
    print("PDF ROUTE: PASSED")


def main():
    if len(sys.argv) != 2:
        raise ValueError(
            "Usage: python -m "
            "scripts.test_agent_regression "
            "<pdf_path>"
        )

    pdf_path = sys.argv[1]

    test_general_route()

    test_research_route()

    test_pdf_route(
        pdf_path
    )

    print()
    print("=" * 70)
    print("ALL REGRESSION TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()