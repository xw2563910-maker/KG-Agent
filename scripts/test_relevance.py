from evidence.relevance import rank_papers


def main():
    papers = [
        {
            "title": (
                "Graph Neural Networks "
                "for Traffic Forecasting"
            ),
            "abstract": (
                "This paper studies graph neural networks "
                "for traffic prediction and transportation "
                "systems."
            ),
        },
        {
            "title": (
                "Empathetic Response Generation "
                "with Reinforcement Learning"
            ),
            "abstract": (
                "This paper studies empathetic response "
                "generation and uses reinforcement learning "
                "to improve response quality."
            ),
        },
        {
            "title": (
                "Empathetic Dialogue Systems"
            ),
            "abstract": (
                "We investigate empathetic response generation "
                "for open-domain dialogue systems and study "
                "methods for generating emotionally appropriate "
                "responses."
            ),
        },
    ]

    query = (
        "empathetic response generation"
    )

    ranked_papers = rank_papers(
        papers,
        query,
    )

    print("Relevance ranking:")

    for index, paper in enumerate(
        ranked_papers,
        start=1,
    ):
        print(
            f"{index}. "
            f"{paper['title']}"
        )

        print(
            f"   score: "
            f"{paper['relevance_score']}"
        )


if __name__ == "__main__":
    main()