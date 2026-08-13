from typing import Any, Literal

from llm.client import chat_json


KGEntityType = Literal[
    "paper",
    "method",
    "dataset",
    "task",
]


KG_QUERY_PLAN_SYSTEM_PROMPT = """
You are the knowledge graph query planner of a scientific
research assistant.

Convert the user's knowledge graph question into a structured
query plan.

The scientific knowledge graph contains these entities:

- Paper
- Author
- Venue
- Method
- Dataset
- Task

The current query interface supports exactly these entity types:

- paper
- method
- dataset
- task

Return exactly this JSON structure:

{
    "entity_type": "paper | method | dataset | task",
    "entity_value": "entity name or identifier"
}

Rules:

1. paper
   Use when the user asks for structured knowledge about a
   specific paper.

2. method
   Use when the user asks which papers use a particular method,
   model, algorithm, or framework.

3. dataset
   Use when the user asks which papers use or evaluate on a
   particular dataset, corpus, or benchmark.

4. task
   Use when the user asks which papers address a particular
   scientific or machine learning task.

5. Preserve the entity name from the user's question.

6. Do not invent an entity that is not mentioned by the user.

7. Do not generate Cypher.

8. Return JSON only.
""".strip()


def build_kg_query_plan(
    question: str,
) -> dict[str, Any]:
    question = str(
        question or ""
    ).strip()

    if not question:
        raise ValueError(
            "KG question cannot be empty."
        )

    plan = chat_json(
        prompt=question,
        system_prompt=(
            KG_QUERY_PLAN_SYSTEM_PROMPT
        ),
    )

    entity_type = str(
        plan.get(
            "entity_type",
            "",
        )
    ).strip().lower()

    entity_value = str(
        plan.get(
            "entity_value",
            "",
        )
    ).strip()

    valid_entity_types = {
        "paper",
        "method",
        "dataset",
        "task",
    }

    if entity_type not in valid_entity_types:
        raise RuntimeError(
            "KG query planner returned "
            f"an invalid entity_type: "
            f"{entity_type}"
        )

    if not entity_value:
        raise RuntimeError(
            "KG query planner returned "
            "an empty entity_value."
        )

    return {
        "entity_type": entity_type,
        "entity_value": entity_value,
    }