import json
from typing import Any

from llm.client import chat


KG_ANSWER_SYSTEM_PROMPT = """
You are the knowledge graph answer component of a scientific
research assistant.

Answer the user's question using only the knowledge graph
evidence provided in the prompt.

Rules:

- Use only the provided knowledge graph evidence.
- Do not add facts from memory or outside knowledge.
- Do not fabricate papers, authors, venues, methods, datasets,
  tasks, years, identifiers, or relationships.
- If a fact is not present in the evidence, do not claim it.
- Clearly distinguish what is present in the current knowledge graph.
- Do not imply that the current knowledge graph contains the complete
  scientific literature.
- Keep the answer concise and directly relevant to the question.
- Answer in the same language as the user's question whenever
  practical.
""".strip()


def _answer_not_found(
    entity_type: str,
    entity_value: str,
) -> str:
    return (
        "当前知识图谱中没有找到与 "
        f"{entity_type} "
        f"“{entity_value}” "
        "匹配的记录。"
    )


def _answer_ambiguous(
    entity_value: str,
    results: list[dict[str, Any]],
) -> str:
    lines = [
        (
            "当前知识图谱中找到多个可能匹配"
            f"“{entity_value}”的论文："
        )
    ]

    for index, item in enumerate(
        results,
        start=1,
    ):
        title = (
            item.get("title")
            or "Unknown title"
        )

        year = item.get("year")

        openalex_id = (
            item.get("openalex_id")
            or "Unknown OpenAlex ID"
        )

        if year is None:
            title_line = (
                f"{index}. {title}"
            )
        else:
            title_line = (
                f"{index}. {title} "
                f"({year})"
            )

        lines.append(
            title_line
        )

        lines.append(
            f"   OpenAlex ID: "
            f"{openalex_id}"
        )

    lines.append(
        "请提供更完整的论文标题或 "
        "OpenAlex ID，以便确定具体论文。"
    )

    return "\n".join(
        lines
    )


def answer_kg_question(
    question: str,
    retrieval_result: dict[str, Any],
) -> str:
    question = str(
        question or ""
    ).strip()

    if not question:
        raise ValueError(
            "Question cannot be empty."
        )

    if not isinstance(
        retrieval_result,
        dict,
    ):
        raise TypeError(
            "KG retrieval result must "
            "be a dictionary."
        )

    status = str(
        retrieval_result.get(
            "status",
            "",
        )
    ).strip().lower()

    entity_type = str(
        retrieval_result.get(
            "entity_type",
            "",
        )
    ).strip()

    entity_value = str(
        retrieval_result.get(
            "entity_value",
            "",
        )
    ).strip()

    results = retrieval_result.get(
        "results",
        [],
    )

    if not isinstance(
        results,
        list,
    ):
        raise TypeError(
            "KG retrieval results must "
            "be a list."
        )

    if status == "not_found":
        return _answer_not_found(
            entity_type,
            entity_value,
        )

    if status == "ambiguous":
        return _answer_ambiguous(
            entity_value,
            results,
        )

    if status != "ok":
        raise ValueError(
            "Unsupported KG retrieval "
            f"status: {status}"
        )

    if not results:
        raise RuntimeError(
            "KG retrieval status is 'ok' "
            "but results are empty."
        )

    evidence_json = json.dumps(
        results,
        ensure_ascii=False,
        indent=2,
    )

    prompt = f"""
User question:
{question}

Knowledge graph query:
- Entity type: {entity_type}
- Entity value: {entity_value}

Knowledge graph evidence:
{evidence_json}

Answer the user's question using only the knowledge graph
evidence above.
""".strip()

    answer = chat(
        prompt,
        system_prompt=(
            KG_ANSWER_SYSTEM_PROMPT
        ),
    )

    answer = answer.strip()

    if not answer:
        raise RuntimeError(
            "KG answer model returned "
            "an empty answer."
        )

    return answer