import re
import unicodedata


def normalize_entity_name(
    name: str,
) -> str:
    if not isinstance(name, str):
        raise TypeError(
            "Entity name must be a string."
        )

    normalized = unicodedata.normalize(
        "NFKC",
        name,
    )

    normalized = normalized.strip()

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    )

    normalized = normalized.casefold()

    if not normalized:
        raise ValueError(
            "Entity name cannot be empty."
        )

    return normalized