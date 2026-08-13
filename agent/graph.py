from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from agent.nodes import (
    build_search_plan_node,
    evidence_quality_node,
    general_answer_node,
    hybrid_answer_node,
    kg_answer_node,
    kg_auto_ingestion_node,
    kg_query_plan_node,
    kg_retrieval_node,
    paper_search_node,
    pdf_answer_node,
    pdf_retrieval_node,
    planner_node,
    relevance_ranking_node,
    research_answer_node,
    route_after_kg_retrieval,
    route_after_pdf_retrieval,
    route_question,
)
from agent.state import AgentState


def create_agent_graph():
    builder = StateGraph(
        AgentState
    )

    # Planner
    builder.add_node(
        "planner",
        planner_node,
    )

    # General route
    builder.add_node(
        "general_answer",
        general_answer_node,
    )

    # Research route
    builder.add_node(
        "build_search_plan",
        build_search_plan_node,
    )
    builder.add_node(
        "paper_search",
        paper_search_node,
    )
    builder.add_node(
        "evidence_quality",
        evidence_quality_node,
    )
    builder.add_node(
        "relevance_ranking",
        relevance_ranking_node,
    )
    builder.add_node(
        "kg_auto_ingestion",
        kg_auto_ingestion_node,
    )
    builder.add_node(
        "research_answer",
        research_answer_node,
    )

    # PDF RAG route
    builder.add_node(
        "pdf_retrieval",
        pdf_retrieval_node,
    )
    builder.add_node(
        "pdf_answer",
        pdf_answer_node,
    )

    # Knowledge-graph route
    builder.add_node(
        "kg_query_plan",
        kg_query_plan_node,
    )
    builder.add_node(
        "kg_retrieval",
        kg_retrieval_node,
    )
    builder.add_node(
        "kg_answer",
        kg_answer_node,
    )

    # Hybrid KG + PDF RAG route
    builder.add_node(
        "hybrid_answer",
        hybrid_answer_node,
    )

    builder.add_edge(
        START,
        "planner",
    )

    builder.add_conditional_edges(
        "planner",
        route_question,
        {
            "general": "general_answer",
            "research": "build_search_plan",
            "pdf": "pdf_retrieval",
            "kg": "kg_query_plan",
            "hybrid": "kg_query_plan",
        },
    )

    # General pipeline
    builder.add_edge(
        "general_answer",
        END,
    )

    # Research pipeline
    builder.add_edge(
        "build_search_plan",
        "paper_search",
    )
    builder.add_edge(
        "paper_search",
        "evidence_quality",
    )
    builder.add_edge(
        "evidence_quality",
        "relevance_ranking",
    )
    builder.add_edge(
        "relevance_ranking",
        "kg_auto_ingestion",
    )
    builder.add_edge(
        "kg_auto_ingestion",
        "research_answer",
    )
    builder.add_edge(
        "research_answer",
        END,
    )

    # Shared KG pipeline. The post-retrieval router decides whether
    # this is a pure KG query or the first half of a hybrid query.
    builder.add_edge(
        "kg_query_plan",
        "kg_retrieval",
    )
    builder.add_conditional_edges(
        "kg_retrieval",
        route_after_kg_retrieval,
        {
            "kg": "kg_answer",
            "hybrid": "pdf_retrieval",
        },
    )
    builder.add_edge(
        "kg_answer",
        END,
    )

    # Shared PDF pipeline. Pure PDF questions go to the PDF answer
    # node; hybrid questions continue to the fusion answer node.
    builder.add_conditional_edges(
        "pdf_retrieval",
        route_after_pdf_retrieval,
        {
            "pdf": "pdf_answer",
            "hybrid": "hybrid_answer",
        },
    )
    builder.add_edge(
        "pdf_answer",
        END,
    )
    builder.add_edge(
        "hybrid_answer",
        END,
    )

    return builder.compile()


agent_graph = create_agent_graph()


def run_agent(
    question: str,
    pdf_path: str | None = None,
) -> str:
    if not question.strip():
        raise ValueError(
            "Question cannot be empty."
        )

    input_state = {
        "question": question
    }

    if pdf_path is not None:
        pdf_path = pdf_path.strip()

        if not pdf_path:
            raise ValueError(
                "pdf_path cannot be empty."
            )

        input_state[
            "pdf_path"
        ] = pdf_path

    result = agent_graph.invoke(
        input_state
    )

    answer = result.get(
        "answer"
    )

    if not answer:
        raise RuntimeError(
            "Agent finished without "
            "producing an answer."
        )

    return answer
