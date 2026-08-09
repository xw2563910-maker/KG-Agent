from typing import Literal, NotRequired, TypedDict
from typing import Any


Route = Literal["general", "research"]


class AgentState(TypedDict):
    question: str
    route: NotRequired[Route]
    search_query: NotRequired[str]
    papers: NotRequired[list[dict[str, Any]]]
    answer: NotRequired[str]