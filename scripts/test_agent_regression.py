import sys
from pathlib import Path

from agent.graph import agent_graph


def _invoke(question: str, pdf_path: str | None = None) -> dict:
    state = {
        "question": question,
    }

    if pdf_path is not None:
        state["pdf_path"] = pdf_path

    return agent_graph.invoke(state)


def _assert_answer(result: dict, route: str) -> None:
    actual_route = result.get("route")
    answer = result.get("answer")

    if actual_route != route:
        raise AssertionError(
            f"Expected route {route!r}, got {actual_route!r}."
        )

    if not answer:
        raise AssertionError(
            f"{route} route returned no answer."
        )


def test_general_route() -> None:
    print("\n" + "=" * 70)
    print("TEST 1: GENERAL ROUTE")
    print("=" * 70)

    result = _invoke(
        "What is overfitting in machine learning?"
    )

    _assert_answer(result, "general")
    print("GENERAL ROUTE: PASSED")


def test_research_route() -> None:
    print("\n" + "=" * 70)
    print("TEST 2: RESEARCH ROUTE + KG AUTO-INGESTION")
    print("=" * 70)

    result = _invoke(
        "请分析近三年共情回复生成的研究趋势"
    )

    _assert_answer(result, "research")

    candidates = result.get("candidate_papers", [])
    quality = result.get("quality_papers", [])
    selected = result.get("papers", [])
    report = result.get("kg_ingestion_report")

    if not candidates or not quality or not selected:
        raise AssertionError(
            "Research evidence pipeline is incomplete."
        )

    if not report:
        raise AssertionError(
            "Research route produced no KG ingestion report."
        )

    if report.get("total") != len(selected):
        raise AssertionError(
            "KG ingestion report total does not match selected papers."
        )

    # Semantic extraction is best-effort by design. Metadata ingestion
    # should process all selected papers when Neo4j is available.
    if report.get("metadata_ingested") not in {0, len(selected)}:
        raise AssertionError(
            "Research KG metadata ingestion was only partially completed."
        )

    print(
        f"Evidence: {len(candidates)} candidates -> "
        f"{len(quality)} quality -> {len(selected)} selected"
    )
    print(
        "KG ingestion: "
        f"metadata={report.get('metadata_ingested')}, "
        f"semantic={report.get('semantic_ingested')}, "
        f"errors={len(report.get('errors', []))}"
    )
    print("RESEARCH ROUTE: PASSED")


def test_pdf_route(pdf_path: str) -> None:
    print("\n" + "=" * 70)
    print("TEST 3: PDF ROUTE")
    print("=" * 70)

    result = _invoke(
        "What reinforcement learning algorithm does EmpRL use?",
        pdf_path=pdf_path,
    )

    _assert_answer(result, "pdf")

    if not result.get("retrieved_chunks"):
        raise AssertionError(
            "PDF route retrieved no chunks."
        )

    print(
        f"Retrieved chunks: {len(result['retrieved_chunks'])}"
    )
    print("PDF ROUTE: PASSED")


def test_kg_route() -> None:
    print("\n" + "=" * 70)
    print("TEST 4: KG ROUTE")
    print("=" * 70)

    result = _invoke(
        "请介绍知识图谱中的 CEM 论文。"
    )

    _assert_answer(result, "kg")

    if not result.get("kg_query_plan"):
        raise AssertionError(
            "KG route produced no query plan."
        )

    retrieval = result.get("kg_retrieval_result")

    if not retrieval:
        raise AssertionError(
            "KG route produced no retrieval result."
        )

    print(
        f"KG status: {retrieval.get('status')}, "
        f"results={len(retrieval.get('results', []))}"
    )
    print("KG ROUTE: PASSED")


def test_hybrid_route(pdf_path: str) -> None:
    print("\n" + "=" * 70)
    print("TEST 5: HYBRID KG + PDF ROUTE")
    print("=" * 70)

    result = _invoke(
        (
            "结合这篇 PDF 和知识图谱中的 CEM 论文信息，"
            "请区分两个证据来源分别支持了哪些结论。"
        ),
        pdf_path=pdf_path,
    )

    _assert_answer(result, "hybrid")

    if not result.get("kg_retrieval_result"):
        raise AssertionError(
            "Hybrid route produced no KG evidence."
        )

    if not result.get("retrieved_chunks"):
        raise AssertionError(
            "Hybrid route produced no PDF evidence."
        )

    print(
        "Hybrid evidence: "
        f"KG={len(result['kg_retrieval_result'].get('results', []))}, "
        f"PDF chunks={len(result['retrieved_chunks'])}"
    )
    print("HYBRID ROUTE: PASSED")


def main() -> None:
    if len(sys.argv) != 2:
        raise ValueError(
            "Usage: python -m scripts.test_agent_regression "
            '"<pdf_path>"'
        )

    pdf_path = Path(sys.argv[1])

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF does not exist: {pdf_path}"
        )

    test_general_route()
    test_research_route()
    test_pdf_route(str(pdf_path))
    test_kg_route()
    test_hybrid_route(str(pdf_path))

    print("\n" + "=" * 70)
    print("ALL 5 AGENT ROUTES: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
