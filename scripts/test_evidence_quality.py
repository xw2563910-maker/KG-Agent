from evidence.quality import filter_papers


def main():
    papers = [
        {
            "openalex_id": "W1",
            "title": "Good Paper",
            "doi": "https://doi.org/10.1/good",
            "abstract": (
                "This paper investigates empathetic response generation "
                "using knowledge-enhanced neural models. The study explores "
                "how external commonsense knowledge can improve emotional "
                "understanding and response generation in dialogue systems. "
                "The proposed approach combines contextual representations "
                "with structured knowledge and evaluates the method on "
                "multiple dialogue benchmarks. Experimental results indicate "
                "improvements in response relevance, emotional appropriateness, "
                "and perceived empathy. Human evaluation is also conducted "
                "to examine whether generated responses better understand "
                "the emotional situations expressed by users."
            ),
        },
        {
            "openalex_id": "W2",
            "title": "Bad Abstract Paper",
            "doi": "https://doi.org/10.1/bad",
            "abstract": (
                "John Smith, Jane Doe. "
                "Proceedings of ACL. 2025."
            ),
        },
        {
            "openalex_id": "W3",
            "title": "Missing Abstract Paper",
            "doi": "https://doi.org/10.1/missing",
            "abstract": None,
        },
        {
            "openalex_id": "W4",
            "title": "Duplicate Good Paper",
            "doi": "10.1/good",
            "abstract": (
                "This record represents another version of the same "
                "academic paper and contains enough words to pass a "
                "simple abstract length test. However, because its DOI "
                "is identical to the first paper after normalization, "
                "the evidence filter should identify this record as "
                "a duplicate and reject it rather than selecting the "
                "same research work twice in the final evidence set."
            ),
        },
    ]

    selected, rejected = filter_papers(
        papers,
        limit=5,
    )

    print("Selected papers:")

    for paper in selected:
        print(
            f"  + {paper['title']}"
        )

    print("\nRejected papers:")

    for item in rejected:
        print(
            f"  - {item['title']}: "
            f"{item['reason']}"
        )


if __name__ == "__main__":
    main()