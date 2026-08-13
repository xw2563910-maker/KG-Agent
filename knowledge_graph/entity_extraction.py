from typing import Any

from llm.client import chat_json


ENTITY_EXTRACTION_SYSTEM_PROMPT = """
You are a scientific information extraction system.

Your task is to extract structured scientific entities from
a paper title and abstract.

Extract only the following entity types:

1. methods
   Algorithms, models, learning methods, frameworks, or
   technical approaches that the paper actually uses,
   proposes, or studies.

2. datasets
   Named datasets, benchmarks, or corpora that the paper
   explicitly uses or evaluates on.

3. tasks
   Scientific or machine learning tasks that the paper
   explicitly addresses.

Rules:

- Use only information explicitly supported by the title
  or abstract.
- Do not infer entities that are not stated.
- Do not invent missing information.
- Prefer specific entity names over vague descriptions.
- Do not include authors, venues, metrics, organizations,
  or generic scientific terms.
- Remove duplicate entities.
- Preserve standard names and acronyms when possible.
- If no entity of a type can be identified, return an
  empty list.
- Return JSON only.

Required JSON format:

{
  "methods": ["method 1", "method 2"],
  "datasets": ["dataset 1", "dataset 2"],
  "tasks": ["task 1", "task 2"]
}
"""


def _normalize_entity_list(
    value: Any,
) -> list[str]:
    if not isinstance(value, list):
        return []

    entities = []
    seen = set()

    for item in value:
        if not isinstance(item, str):
            continue

        entity = item.strip()

        if not entity:
            continue

        key = entity.casefold()

        if key in seen:
            continue

        seen.add(key)
        entities.append(entity)

    return entities


def extract_scientific_entities(
    paper: dict[str, Any],
) -> dict[str, list[str]]:
    title = str(
        paper.get("title")
        or ""
    ).strip()

    abstract = str(
        paper.get("abstract")
        or ""
    ).strip()

    if not title and not abstract:
        raise ValueError(
            "Paper title and abstract "
            "cannot both be empty."
        )

    prompt = f"""
Paper title:
{title}

Paper abstract:
{abstract}

Extract the scientific entities.
""".strip()

    result = chat_json(
        prompt=prompt,
        system_prompt=(
            ENTITY_EXTRACTION_SYSTEM_PROMPT
        ),
    )

    methods = _normalize_entity_list(
        result.get("methods")
    )

    datasets = _normalize_entity_list(
        result.get("datasets")
    )

    tasks = _normalize_entity_list(
        result.get("tasks")
    )

    return {
        "methods": methods,
        "datasets": datasets,
        "tasks": tasks,
    }