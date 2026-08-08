from typing import Literal, NotRequired, TypedDict


Route = Literal["general", "research"]


class AgentState(TypedDict):
    question: str
    route: NotRequired[Route]
    answer: NotRequired[str]