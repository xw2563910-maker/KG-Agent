from knowledge_graph.normalization import (
    normalize_entity_name,
)


def main():
    test_cases = [
        (
            "CEM",
            "cem",
        ),
        (
            "cem",
            "cem",
        ),
        (
            " CEM ",
            "cem",
        ),
        (
            "EMPATHETICDIALOGUES",
            "empatheticdialogues",
        ),
        (
            "EmpatheticDialogues",
            "empatheticdialogues",
        ),
        (
            "empatheticdialogues",
            "empatheticdialogues",
        ),
        (
            "Empathetic Response Generation",
            "empathetic response generation",
        ),
        (
            "  Empathetic   Response   Generation  ",
            "empathetic response generation",
        ),
    ]

    for original, expected in test_cases:
        actual = normalize_entity_name(
            original
        )

        print(
            f"{original!r}"
            f" -> "
            f"{actual!r}"
        )

        if actual != expected:
            raise AssertionError(
                f"Expected {expected!r}, "
                f"got {actual!r}"
            )

    print()

    print(
        "Entity normalization: PASSED"
    )


if __name__ == "__main__":
    main()