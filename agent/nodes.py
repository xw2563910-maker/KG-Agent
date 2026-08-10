from agent.state import AgentState, Route
from llm.client import chat, chat_json
from tools.paper_search import search_papers
from datetime import date
from evidence.quality import filter_papers
from evidence.relevance import rank_papers
from evidence.semantic_relevance import (
    rank_papers_semantically,
)

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

Answer the user's research question using only the retrieved academic
evidence provided in the prompt.

Requirements:

- Use the retrieved paper metadata and abstracts as the primary evidence.
- Do not fabricate papers, authors, venues, methods, datasets,
  experimental results, or citations.
- Do not claim that a paper proposes a method unless that claim is
  supported by the provided title or abstract.
- Clearly distinguish evidence-supported conclusions from reasonable
  inference.
- If the retrieved evidence is limited, incomplete, or weakly related
  to the user's question, state that limitation explicitly.
- Respect the search time range provided in the prompt and do not
  reinterpret relative expressions such as "recent three years".
""".strip()

SEARCH_PLAN_SYSTEM_PROMPT = """
You are the search planner of a scientific research assistant.

Convert the user's research question into a structured academic
paper search plan.

You must return a JSON object with exactly these fields:

{
    "search_query": "English academic search keywords",
    "from_year": null,
    "to_year": null,
    "limit": 5
}

Rules:

1. search_query
   - Use concise English academic keywords.
   - Preserve all important domain-specific concepts from the
     user's question.
   - Do not generalize a specific research topic into a broader one.
   - Translate technical terms into their standard academic English
     expressions whenever possible.
   - Remove only unnecessary natural-language instructions such as
     "please analyze", "find papers", or "research trends".
   - The search query should represent the research topic itself,
     not the requested action.

2. Semantic preservation
   - "共情回复生成" should be preserved as
     "empathetic response generation".
   - "知识图谱推荐系统" should preserve both
     "knowledge graph" and "recommendation".
   - Never remove a concept that changes the research domain.
   - If a user term appears ambiguous or possibly misspelled,
     preserve its likely domain meaning rather than replacing it
     with a broader unrelated concept.

3. from_year and to_year
   - If the user provides an explicit year range, extract it.
   - If the user says "recent N years", calculate the range using
     the current year and the previous N-1 calendar years.
   - If no time constraint is specified, return null for both.

4. limit
   - Default to 5.
   - If the user explicitly requests a number of papers, use it.
   - The value must be between 1 and 10.

Return JSON only.
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


def research_answer_node(
    state: AgentState,
) -> dict:
    print(
        "[LangGraph] Enter research answer node"
    )

    question = state["question"]

    papers = state.get(
        "papers",
        [],
    )

    if not papers:
        raise RuntimeError(
            "No papers found in state "
            "for research answer."
        )

    from_year = state.get(
        "from_year"
    )

    to_year = state.get(
        "to_year"
    )

    search_query = state.get(
        "search_query",
        "",
    )

    candidate_papers = state.get(
        "candidate_papers",
        [],
    )

    quality_papers = state.get(
        "quality_papers",
        [],
    )

    candidate_count = len(
        candidate_papers
    )

    quality_count = len(
        quality_papers
    )

    selected_count = len(
        papers
    )

    print(
        "[LangGraph] Evidence pipeline: "
        f"{candidate_count} candidates -> "
        f"{quality_count} quality -> "
        f"{selected_count} selected"
    )

    paper_lines = []

    for index, paper in enumerate(
        papers,
        start=1,
    ):
        authors = ", ".join(
            paper.get(
                "authors",
                [],
            )
        )

        abstract = (
            paper.get("abstract")
            or "No abstract available."
        )

        paper_text = f"""
Paper {index}

Title:
{paper.get("title")}

Year:
{paper.get("year")}

Authors:
{authors}

Venue:
{paper.get("venue")}

DOI:
{paper.get("doi")}

Citations:
{paper.get("cited_by_count")}

Abstract:
{abstract}
""".strip()

        paper_lines.append(
            paper_text
        )

    papers_context = "\n\n".join(
        paper_lines
    )

    if (
        from_year is not None
        and to_year is not None
    ):
        time_range = (
            f"{from_year}-{to_year}"
        )
    else:
        time_range = (
            "No explicit time restriction"
        )

    prompt = f"""
User question:
{question}

Search plan:
- Search query: {search_query}
- Search time range: {time_range}

Evidence pipeline:
- Candidate papers retrieved: {candidate_count}
- Papers passed evidence quality filtering: {quality_count}
- Final papers selected after semantic ranking: {selected_count}

Important:
The search time range above has already been determined by the
Search Planner. Do not reinterpret it.

The evidence pipeline above describes three different stages:
candidate retrieval, evidence quality filtering, and semantic
ranking. Do not confuse the final selected evidence papers with
the total number of candidate papers originally retrieved.

When describing the literature retrieval process, report these
stages accurately. For example, if 15 candidate papers were
retrieved, 14 passed quality filtering, and 5 were selected after
semantic ranking, do not simply say that only 5 papers were
retrieved.

Final selected academic evidence:

{papers_context}

Please answer the user's research question based on the final
selected academic evidence above.

Requirements:

1. Base claims about individual papers primarily on their provided
   titles and abstracts.

2. Summarize the major research directions or findings supported
   by the final selected evidence sample.

3. Clearly distinguish:
   - evidence directly supported by the selected papers
   - broader inference based on the selected sample

4. When describing the retrieval process, clearly distinguish:
   - candidate papers retrieved
   - papers remaining after evidence quality filtering
   - final papers selected after semantic ranking

5. Do not say that only the final selected papers were "retrieved"
   if more candidate papers were initially retrieved.

6. If the actual publication years of the selected papers do not
   cover every year in the search time range, state this accurately.

7. Do not introduce years outside the specified search time range
   when discussing missing coverage.

8. Do not fabricate papers, methods, datasets, results, citations,
   or other academic evidence.

9. Explicitly state that conclusions are based on the final selected
   evidence sample when the sample size is limited.

10. Do not claim that the selected sample represents the complete
    literature or the definitive research trends of the entire field
    unless the provided evidence is sufficient to support that claim.
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


def build_search_plan_node(
    state: AgentState,
) -> dict:
    print("[LangGraph] Enter search planner node")

    question = state["question"]

    current_year = date.today().year

    prompt = f"""
Current year: {current_year}

User research question:
{question}

Generate the academic paper search plan.
""".strip()

    plan = chat_json(
        prompt,
        system_prompt=SEARCH_PLAN_SYSTEM_PROMPT,
    )

    search_query = str(
        plan.get("search_query", "")
    ).strip()

    if not search_query:
        raise RuntimeError(
            "Search planner returned an empty search_query."
        )

    from_year = plan.get("from_year")
    to_year = plan.get("to_year")
    limit = plan.get("limit", 5)

    if from_year is not None:
        try:
            from_year = int(from_year)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Invalid from_year: {from_year}"
            ) from exc

    if to_year is not None:
        try:
            to_year = int(to_year)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Invalid to_year: {to_year}"
            ) from exc

    if (from_year is None) != (to_year is None):
        raise RuntimeError(
            "Search planner must provide both "
            "from_year and to_year, or neither."
        )

    if (
        from_year is not None
        and to_year is not None
        and from_year > to_year
    ):
        raise RuntimeError(
            "Search planner returned an invalid year range."
        )

    try:
        limit = int(limit)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Invalid search limit: {limit}"
        ) from exc

    if not 1 <= limit <= 10:
        raise RuntimeError(
            "Search planner limit must be between 1 and 10."
        )

    print("[LangGraph] Search plan:")
    print(f"  query: {search_query}")
    print(f"  from_year: {from_year}")
    print(f"  to_year: {to_year}")
    print(f"  limit: {limit}")

    return {
        "search_query": search_query,
        "from_year": from_year,
        "to_year": to_year,
        "search_limit": limit,
    }


def paper_search_node(state: AgentState) -> dict:
    print("[LangGraph] Enter paper search node")

    search_query = state.get("search_query")

    if not search_query:
        raise RuntimeError(
            "No search_query found in state."
        )

    from_year = state.get("from_year")
    to_year = state.get("to_year")

    search_limit = state.get(
        "search_limit",
        5,
    )

    candidate_limit = min(
        search_limit * 3,
        25,
    )

    candidate_papers = search_papers(
        query=search_query,
        limit=candidate_limit,
        from_year=from_year,
        to_year=to_year,
        require_abstract=True,
    )

    print(
        f"[LangGraph] Retrieved "
        f"{len(candidate_papers)} "
        f"candidate papers"
    )

    return {
        "candidate_papers": candidate_papers
    }


def evidence_quality_node(
    state: AgentState,
) -> dict:
    print(
        "[LangGraph] Enter evidence quality node"
    )

    candidate_papers = state.get(
        "candidate_papers",
        [],
    )

    if not candidate_papers:
        raise RuntimeError(
            "No candidate papers found in state."
        )

    quality_papers, rejected_papers = (
        filter_papers(
            candidate_papers
        )
    )

    print(
        "[LangGraph] Evidence quality:"
    )

    print(
        f"  candidates: "
        f"{len(candidate_papers)}"
    )

    print(
        f"  rejected: "
        f"{len(rejected_papers)}"
    )

    print(
        f"  accepted: "
        f"{len(quality_papers)}"
    )

    for rejected in rejected_papers:
        print(
            f"  - rejected: "
            f"{rejected['title']} "
            f"({rejected['reason']})"
        )

    if not quality_papers:
        raise RuntimeError(
            "No papers remained after "
            "evidence quality filtering."
        )

    return {
        "quality_papers": quality_papers
    }


def relevance_ranking_node(
    state: AgentState,
) -> dict:
    print(
        "[LangGraph] Enter relevance ranking node"
    )

    quality_papers = state.get(
        "quality_papers",
        [],
    )

    if not quality_papers:
        raise RuntimeError(
            "No quality papers found in state."
        )

    search_query = state.get(
        "search_query",
        "",
    )

    if not search_query:
        raise RuntimeError(
            "No search_query found in state "
            "for relevance ranking."
        )

    search_limit = state.get(
        "search_limit",
        5,
    )

    ranked_papers = (
        rank_papers_semantically(
            quality_papers,
            search_query,
        )
    )

    selected_papers = ranked_papers[
        :search_limit
    ]

    print(
        "[LangGraph] Semantic relevance ranking:"
    )

    print(
        f"  quality papers: "
        f"{len(quality_papers)}"
    )

    print(
        f"  selected: "
        f"{len(selected_papers)}"
    )

    for index, paper in enumerate(
        selected_papers,
        start=1,
    ):
        print(
            f"  {index}. "
            f"{paper.get('title')} "
            f"(semantic_score="
            f"{paper.get('semantic_score')})"
        )

    if not selected_papers:
        raise RuntimeError(
            "No papers remained after "
            "semantic relevance ranking."
        )

    return {
        "papers": selected_papers
    }