from typing import Any, Literal, NotRequired, TypedDict


Route = Literal["general", "research"]


class AgentState(TypedDict):
    question: str

    route: NotRequired[Route]

    search_query: NotRequired[str]
    from_year: NotRequired[int]
    to_year: NotRequired[int]
    search_limit: NotRequired[int]

    candidate_papers: NotRequired[
        list[dict[str, Any]]
    ]

    quality_papers: NotRequired[
        list[dict[str, Any]]
    ]

    papers: NotRequired[
        list[dict[str, Any]]
    ]

    answer: NotRequired[str]