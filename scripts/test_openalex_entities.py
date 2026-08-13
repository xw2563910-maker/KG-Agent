from tools.paper_search import search_papers


def main():
    papers = search_papers(
        query="empathetic response generation",
        limit=1,
        require_abstract=True,
    )

    if not papers:
        raise RuntimeError(
            "OpenAlex returned no papers."
        )

    paper = papers[0]

    print("=" * 70)
    print("PAPER")
    print("=" * 70)

    print(
        f"OpenAlex ID: "
        f"{paper.get('openalex_id')}"
    )

    print(
        f"Title: "
        f"{paper.get('title')}"
    )

    print(
        f"Year: "
        f"{paper.get('year')}"
    )

    print()

    print("=" * 70)
    print("AUTHORS")
    print("=" * 70)

    authors = paper.get(
        "authors",
        [],
    )

    print(
        f"Original authors: {authors}"
    )

    print()

    author_entities = paper.get(
        "author_entities",
        [],
    )

    if not author_entities:
        raise AssertionError(
            "No structured author entities found."
        )

    for index, author in enumerate(
        author_entities,
        start=1,
    ):
        print(
            f"Author {index}:"
        )

        print(
            f"  OpenAlex ID: "
            f"{author.get('openalex_id')}"
        )

        print(
            f"  Name: "
            f"{author.get('name')}"
        )

    print()

    print("=" * 70)
    print("VENUE")
    print("=" * 70)

    print(
        f"Original venue: "
        f"{paper.get('venue')}"
    )

    venue_entity = paper.get(
        "venue_entity"
    )

    if venue_entity is None:
        print(
            "Structured venue entity: None"
        )
    else:
        print(
            f"OpenAlex ID: "
            f"{venue_entity.get('openalex_id')}"
        )

        print(
            f"Name: "
            f"{venue_entity.get('name')}"
        )

    print()

    print("=" * 70)
    print("VALIDATION")
    print("=" * 70)

    paper_id = paper.get(
        "openalex_id"
    )

    if not paper_id:
        raise AssertionError(
            "Paper OpenAlex ID is missing."
        )

    if not paper_id.startswith(
        "https://openalex.org/W"
    ):
        raise AssertionError(
            "Paper OpenAlex ID does not "
            "look like a Work ID."
        )

    for author in author_entities:
        author_id = author.get(
            "openalex_id"
        )

        if not author_id:
            raise AssertionError(
                "Author OpenAlex ID is missing."
            )

        if not author_id.startswith(
            "https://openalex.org/A"
        ):
            raise AssertionError(
                "Author OpenAlex ID does not "
                "look like an Author ID."
            )

    if venue_entity is not None:
        venue_id = venue_entity.get(
            "openalex_id"
        )

        if not venue_id:
            raise AssertionError(
                "Venue OpenAlex ID is missing."
            )

        if not venue_id.startswith(
            "https://openalex.org/S"
        ):
            raise AssertionError(
                "Venue OpenAlex ID does not "
                "look like a Source ID."
            )

    print(
        "OpenAlex entity metadata: PASSED"
    )


if __name__ == "__main__":
    main()