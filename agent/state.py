from typing import (
    Any,
    Literal,
    NotRequired,
    TypedDict,
)


Route = Literal[
    "general",
    "research",
    "pdf",
    "kg",
    "hybrid",
]


class AgentState(TypedDict):
    question: str

    route: NotRequired[Route]

    # ============================================================
    # Research Route
    # ============================================================

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

    # ============================================================
    # PDF RAG Route
    # ============================================================

    pdf_path: NotRequired[str]

    retrieved_chunks: NotRequired[
        list[dict[str, Any]]
    ]

    # ============================================================
    # Knowledge Graph Route
    # ============================================================

    kg_query_plan: NotRequired[
        dict[str, Any]
    ]

    kg_retrieval_result: NotRequired[
        dict[str, Any]
    ]

    kg_ingestion_report: NotRequired[
        dict[str, Any]
    ]

    # ============================================================
    # Final Answer
    # ============================================================

    answer: NotRequired[str]
