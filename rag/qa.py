from typing import Any

from llm.client import chat


PDF_RAG_SYSTEM_PROMPT = """
You are a scientific PDF question-answering assistant.

Answer the user's question using only the retrieved PDF
evidence provided in the prompt.

Requirements:

- Base your answer only on the provided evidence.
- Do not use unsupported external knowledge.
- Do not fabricate methods, datasets, results, or citations.
- Cite supporting evidence using labels such as [Evidence 1].
- If the evidence is insufficient to answer the question,
  state that clearly.
- Do not treat retrieval similarity scores as confidence
  probabilities.
- Avoid repeating the same factual claim across
  multiple sentences.
""".strip()


def build_evidence_context(
    retrieved_chunks: list[dict[str, Any]],
) -> str:
    """
    Convert retrieved PDF chunks into structured
    evidence text for the LLM.
    """
    if not retrieved_chunks:
        raise ValueError(
            "Retrieved chunks cannot be empty."
        )

    evidence_blocks = []

    for index, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):
        text = str(
            chunk.get("text") or ""
        ).strip()

        if not text:
            continue

        evidence = f"""
[Evidence {index}]

Source: {chunk.get("source")}
Page: {chunk.get("page_number")}
Chunk: {chunk.get("chunk_index")}

Text:
{text}
""".strip()

        evidence_blocks.append(
            evidence
        )

    if not evidence_blocks:
        raise ValueError(
            "No usable evidence text found."
        )

    return "\n\n".join(
        evidence_blocks
    )


def answer_pdf_question(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
) -> str:
    """
    Answer a question using retrieved PDF evidence.
    """
    question = question.strip()

    if not question:
        raise ValueError(
            "Question cannot be empty."
        )

    evidence_context = build_evidence_context(
        retrieved_chunks
    )

    prompt = f"""
User question:

{question}


Retrieved PDF evidence:

{evidence_context}


Answer the user's question using only the evidence above.

For factual claims, cite the relevant evidence labels,
for example [Evidence 1] or [Evidence 2].
""".strip()

    answer = chat(
        prompt,
        system_prompt=PDF_RAG_SYSTEM_PROMPT,
    )

    if not answer.strip():
        raise RuntimeError(
            "LLM returned an empty PDF RAG answer."
        )

    return answer.strip()