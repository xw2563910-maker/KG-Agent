from tools.paper_search import search_papers


def main():
    papers = search_papers(
        query="knowledge graph recommendation",
        limit=5,
        from_year=2024,
        to_year=2026,
        require_abstract=True,
    )

    print(
        f"Found {len(papers)} papers.\n"
    )

    for index, paper in enumerate(
        papers,
        start=1,
    ):
        print("=" * 80)

        print(
            f"Paper {index}"
        )

        print(
            f"Title: {paper['title']}"
        )

        print(
            f"Year: {paper['year']}"
        )

        print(
            f"Authors: "
            f"{', '.join(paper['authors'])}"
        )

        print(
            f"Venue: {paper['venue']}"
        )

        print(
            f"DOI: {paper['doi']}"
        )

        print(
            f"Citations: "
            f"{paper['cited_by_count']}"
        )

        print("\nAbstract:")

        abstract = paper.get(
            "abstract"
        )

        if abstract:
            print(
                abstract[:500]
            )
        else:
            print(
                "No abstract available."
            )

        print()


if __name__ == "__main__":
    main()