from agent.state import AgentState, Route
from llm.client import chat
from tools.paper_search import search_papers

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

SEARCH_QUERY_SYSTEM_PROMPT = """
You are a search query builder for an academic research assistant.

Convert the user's research question into a short and effective academic paper search query.

Requirements:
- Output only one search query
- Keep it concise
- Use English keywords
- Focus on paper retrieval
- Do not explain anything
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

    question = state["question"]
    papers = state.get("papers", [])

    if not papers:
        raise RuntimeError("No papers found in state for research answer.")

    paper_lines = []

    for index, paper in enumerate(papers, start=1):
        authors = ", ".join(paper.get("authors", []))
        line = (
            f"{index}. Title: {paper.get('title')}; "
            f"Year: {paper.get('year')}; "
            f"Authors: {authors}; "
            f"Venue: {paper.get('venue')}; "
            f"DOI: {paper.get('doi')}; "
            f"Citations: {paper.get('cited_by_count')}"
        )
        paper_lines.append(line)

    papers_context = "\n".join(paper_lines)

    prompt = f"""
User question:
{question}

Retrieved papers:
{papers_context}

Please answer the user's research question based on the retrieved papers.

Requirements:
- Use the retrieved papers as evidence
- Summarize the main research directions or findings
- If the papers are limited, explicitly say the conclusion is based on the retrieved sample
- Do not fabricate papers that are not in the list
""".strip()

    answer = chat(
        prompt,
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


def build_search_query_node(state: AgentState) -> dict:
    print("[LangGraph] Enter build search query node")

    question = state["question"]

    search_query = chat(
        question,
        system_prompt=SEARCH_QUERY_SYSTEM_PROMPT,
    ).strip()

    if not search_query:
        raise RuntimeError("Search query builder returned an empty query.")

    print(f"[LangGraph] Search query: {search_query}")

    return {
        "search_query": search_query
    }


def paper_search_node(state: AgentState) -> dict:
    print("[LangGraph] Enter paper search node")

    search_query = state.get("search_query")

    if not search_query:
        raise RuntimeError("No search_query found in state.")

    papers = search_papers(
        query=search_query,
        limit=5,
    )

    print(f"[LangGraph] Retrieved {len(papers)} papers")

    return {
        "papers": papers
    }