from langgraph.graph import END, START, StateGraph

from agent.nodes import assistant_node
from agent.state import AgentState


def create_agent_graph():
    builder = StateGraph(AgentState)

    builder.add_node(
        "assistant",
        assistant_node,
    )

    builder.add_edge(
        START,
        "assistant",
    )

    builder.add_edge(
        "assistant",
        END,
    )

    return builder.compile()


agent_graph = create_agent_graph()


def run_agent(question: str) -> str:
    result = agent_graph.invoke(
        {
            "question": question,
            "answer": "",
        }
    )

    return result["answer"]