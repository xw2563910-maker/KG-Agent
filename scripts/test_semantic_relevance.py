from evidence.semantic_relevance import (
    rank_papers_semantically,
)


def main():
    query = (
        "empathetic response generation"
    )

    papers = [
        {
            "title": (
                "Graph Neural Networks "
                "for Traffic Forecasting"
            ),
            "abstract": (
                "This paper studies traffic "
                "forecasting using graph neural "
                "networks for transportation systems."
            ),
        },
        {
            "title": (
                "Empathetic Response Generation "
                "with Reinforcement Learning"
            ),
            "abstract": (
                "This work investigates empathetic "
                "response generation and applies "
                "reinforcement learning to improve "
                "the empathy level of generated "
                "dialogue responses."
            ),
        },
        {
            "title": (
                "Emotion Recognition "
                "in Dialogue Systems"
            ),
            "abstract": (
                "This paper studies emotion "
                "recognition and emotional "
                "understanding in conversational "
                "dialogue systems."
            ),
        },
    ]

    ranked_papers = (
        rank_papers_semantically(
            papers,
            query,
        )
    )

    print(
        "Semantic relevance ranking:"
    )

    for index, paper in enumerate(
        ranked_papers,
        start=1,
    ):
        print(
            f"{index}. "
            f"{paper['title']}"
        )

        print(
            f"   semantic_score: "
            f"{paper['semantic_score']}"
        )


if __name__ == "__main__":
    main()