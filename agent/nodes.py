from agent.state import AgentState, Route
from llm.client import chat


PLANNER_SYSTEM_PROMPT = """
You are the routing planner of a scientific research assistant.

Classify the user's question into exactly one of these two categories:

general
- General knowledge questions
- Concept explanations
- Definitions
- Simple comparisons
- Casual questions

research
- Literature review requests
- Research trend analysis
- Paper or method comparisons
- Dataset or experiment analysis
- Scientific discovery tasks
- Questions that should use academic knowledge retrieval

Return ONLY one word:

general

or

research

Do not provide any explanation.
""".strip()


GENERAL_SYSTEM_PROMPT = """
You are a scientific research assistant.

Answer general questions clearly, accurately, and concisely.
Explain technical concepts in an easy-to-understand way.
""".strip()


RESEARCH_SYSTEM_PROMPT = """
You are a scientific research analysis assistant.

Provide a structured academic-style analysis of the user's question.

At this stage, you do not have access to external literature retrieval,
so do not fabricate paper titles, authors, citations, datasets, or
experimental results.

Clearly distinguish established knowledge from claims that would require
external evidence.
""".strip()


def planner_node(state: AgentState) -> dict:
    print("[LangGraph] Enter planner node")

    question = state["question"]

    result = chat(
        question,
        system_prompt=PLANNER_SYSTEM_PROMPT,
    )

    route_text = result.strip().lower()

    if "research" in route_text:
        route: Route = "research"
    elif "general" in route_text:
        route = "general"
    else:
        raise RuntimeError(
            f"Planner returned an invalid route: {result}"
        )

    print(f"[LangGraph] Planner route: {route}")

    return {
        "route": route
    }


def general_answer_node(state: AgentState) -> dict:
    print("[LangGraph] Enter general answer node")

    answer = chat(
        state["question"],
        system_prompt=GENERAL_SYSTEM_PROMPT,
    )

    return {
        "answer": answer
    }


def research_answer_node(state: AgentState) -> dict:
    print("[LangGraph] Enter research answer node")

    answer = chat(
        state["question"],
        system_prompt=RESEARCH_SYSTEM_PROMPT,
    )

    return {
        "answer": answer
    }


def route_question(state: AgentState) -> Route:
    route = state.get("route")

    if route not in ("general", "research"):
        raise RuntimeError(
            f"Invalid route in agent state: {route}"
        )

    return route