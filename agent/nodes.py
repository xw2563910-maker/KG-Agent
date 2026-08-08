from agent.state import AgentState
from llm.client import chat


def assistant_node(state: AgentState) -> dict:
    question = state["question"]

    answer = chat(question)

    return {
        "answer": answer
    }