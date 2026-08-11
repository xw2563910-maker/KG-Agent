from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    build_search_plan_node,
    evidence_quality_node,
    general_answer_node,
    paper_search_node,
    pdf_answer_node,
    pdf_retrieval_node,
    planner_node,
    relevance_ranking_node,
    research_answer_node,
    route_question,
)
from agent.state import AgentState


def create_agent_graph():
    builder = StateGraph(AgentState)

    builder.add_node(
        "planner",
        planner_node,
    )

    builder.add_node(
        "general_answer",
        general_answer_node,
    )

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
        "research_answer",
        research_answer_node,
    )
    builder.add_node(
        "pdf_retrieval",
        pdf_retrieval_node,
    )

    builder.add_node(
        "pdf_answer",
        pdf_answer_node,
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
        },
    )

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
        "research_answer",
    )

    builder.add_edge(
        "general_answer",
        END,
    )

    builder.add_edge(
        "research_answer",
        END,
    )
    builder.add_edge(
        "pdf_retrieval",
        "pdf_answer",
    )

    builder.add_edge(
        "pdf_answer",
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

        input_state["pdf_path"] = pdf_path

    result = agent_graph.invoke(
        input_state
    )

    answer = result.get("answer")

    if not answer:
        raise RuntimeError(
            "Agent finished without "
            "producing an answer."
        )

    return answer