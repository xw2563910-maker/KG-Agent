from knowledge_graph.entity_extraction import (
    extract_scientific_entities,
)
from tools.paper_search import (
    search_papers,
)


def main():
    papers = search_papers(
        query=(
            "CEM Commonsense-Aware "
            "Empathetic Response Generation"
        ),
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
        f"Title: "
        f"{paper.get('title')}"
    )

    print(
        f"OpenAlex ID: "
        f"{paper.get('openalex_id')}"
    )

    print()

    print("=" * 70)
    print("SCIENTIFIC ENTITY EXTRACTION")
    print("=" * 70)

    entities = extract_scientific_entities(
        paper
    )

    print()
    print("Methods:")

    if entities["methods"]:
        for method in entities["methods"]:
            print(
                f"  - {method}"
            )
    else:
        print(
            "  - None"
        )

    print()
    print("Datasets:")

    if entities["datasets"]:
        for dataset in entities["datasets"]:
            print(
                f"  - {dataset}"
            )
    else:
        print(
            "  - None"
        )

    print()
    print("Tasks:")

    if entities["tasks"]:
        for task in entities["tasks"]:
            print(
                f"  - {task}"
            )
    else:
        print(
            "  - None"
        )

    print()

    print("=" * 70)
    print("VALIDATION")
    print("=" * 70)

    required_keys = {
        "methods",
        "datasets",
        "tasks",
    }

    if set(entities.keys()) != required_keys:
        raise AssertionError(
            "Unexpected entity extraction "
            "schema."
        )

    for key in required_keys:
        if not isinstance(
            entities[key],
            list,
        ):
            raise AssertionError(
                f"{key} must be a list."
            )

        for entity in entities[key]:
            if not isinstance(
                entity,
                str,
            ):
                raise AssertionError(
                    f"{key} contains "
                    "a non-string entity."
                )

            if not entity.strip():
                raise AssertionError(
                    f"{key} contains "
                    "an empty entity."
                )

    print(
        "Scientific entity extraction: PASSED"
    )


if __name__ == "__main__":
    main()