from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    general_answer_node,
    planner_node,
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
        "research_answer",
        research_answer_node,
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
            "research": "research_answer",
        },
    )

    builder.add_edge(
        "general_answer",
        END,
    )

    builder.add_edge(
        "research_answer",
        END,
    )

    return builder.compile()


agent_graph = create_agent_graph()


def run_agent(question: str) -> str:
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    result = agent_graph.invoke(
        {
            "question": question,
        }
    )

    answer = result.get("answer")

    if not answer:
        raise RuntimeError(
            "Agent finished without producing an answer."
        )

    return answer