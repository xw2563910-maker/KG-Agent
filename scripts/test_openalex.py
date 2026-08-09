from tools.paper_search import search_papers


def main():
    papers = search_papers(
        query="knowledge graph recommendation",
        limit=3,
    )

    print(f"Found {len(papers)} papers.\n")

    for index, paper in enumerate(papers, start=1):
        print("=" * 70)
        print(f"Paper {index}")
        print(f"Title: {paper['title']}")
        print(f"Year: {paper['year']}")
        print(f"Authors: {', '.join(paper['authors'])}")
        print(f"Venue: {paper['venue']}")
        print(f"DOI: {paper['doi']}")
        print(f"Citations: {paper['cited_by_count']}")


if __name__ == "__main__":
    main()