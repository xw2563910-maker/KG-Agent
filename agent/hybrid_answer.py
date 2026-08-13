import json
from typing import Any

from llm.client import chat


HYBRID_ANSWER_SYSTEM_PROMPT = """
You are the hybrid evidence answer component of a scientific
research assistant.

You are given two possible evidence sources:

1. Knowledge Graph Evidence
   - Structured facts retrieved from Neo4j.
   - May contain papers, authors, venues, methods, datasets,
     tasks, and explicit graph relationships.

2. PDF RAG Evidence
   - Text chunks retrieved from a user-provided PDF through
     semantic vector search.

Answer the user's question using only the evidence provided.

Rules:

- Do not use outside knowledge or memory.
- Do not fabricate papers, authors, venues, methods, datasets,
  tasks, experimental results, citations, or relationships.
- Clearly distinguish facts supported by the knowledge graph
  from facts supported by the PDF.
- If both sources support the same conclusion, say that the
  evidence is consistent across both sources.
- If the two sources provide different or apparently conflicting
  information, describe the difference instead of silently
  resolving it.
- If one source does not contain relevant evidence, state that
  limitation.
- A knowledge graph with status "not_found" contains no matching
  structured evidence for the requested entity.
- A knowledge graph with status "ambiguous" contains multiple
  possible matches. Do not choose one candidate unless the
  evidence clearly identifies it.
- Do not imply that the knowledge graph contains the complete
  scientific literature.
- Do not imply that the retrieved PDF chunks represent the entire
  PDF unless the supplied evidence supports that statement.
- Answer in the same language as the user's question whenever
  practical.
""".strip()


def answer_hybrid_question(
    question: str,
    kg_retrieval_result: dict[str, Any],
    retrieved_chunks: list[dict[str, Any]],
) -> str:
    question = str(
        question or ""
    ).strip()

    if not question:
        raise ValueError(
            "Question cannot be empty."
        )

    if not isinstance(
        kg_retrieval_result,
        dict,
    ):
        raise TypeError(
            "KG retrieval result must "
            "be a dictionary."
        )

    if not isinstance(
        retrieved_chunks,
        list,
    ):
        raise TypeError(
            "Retrieved PDF chunks must "
            "be a list."
        )

    kg_status = str(
        kg_retrieval_result.get(
            "status",
            "",
        )
    ).strip().lower()

    if kg_status not in {
        "ok",
        "not_found",
        "ambiguous",
    }:
        raise ValueError(
            "Unsupported KG retrieval "
            f"status: {kg_status}"
        )

    kg_results = (
        kg_retrieval_result.get(
            "results",
            [],
        )
    )

    if not isinstance(
        kg_results,
        list,
    ):
        raise TypeError(
            "KG retrieval results must "
            "be a list."
        )

    if not retrieved_chunks:
        raise RuntimeError(
            "Hybrid answering requires "
            "retrieved PDF evidence."
        )

    kg_evidence = {
        "entity_type": (
            kg_retrieval_result.get(
                "entity_type"
            )
        ),
        "entity_value": (
            kg_retrieval_result.get(
                "entity_value"
            )
        ),
        "status": kg_status,
        "results": kg_results,
    }

    kg_json = json.dumps(
        kg_evidence,
        ensure_ascii=False,
        indent=2,
    )

    pdf_json = json.dumps(
        retrieved_chunks,
        ensure_ascii=False,
        indent=2,
        default=str,
    )

    prompt = f"""
User question:
{question}

============================================================
KNOWLEDGE GRAPH EVIDENCE
============================================================

{kg_json}

============================================================
PDF RAG EVIDENCE
============================================================

{pdf_json}

============================================================
ANSWER REQUIREMENTS
============================================================

Answer the user's question using only the two evidence sources
above.

In the answer:

1. Explain what the knowledge graph supports.

2. Explain what the retrieved PDF evidence supports.

3. If both evidence sources support the same conclusion, explicitly
   state that they are consistent.

4. If one source lacks relevant evidence, say so.

5. If the evidence sources differ, report the difference instead
   of inventing a reconciliation.

6. Do not introduce information that is absent from both evidence
   sources.
""".strip()

    answer = chat(
        prompt,
        system_prompt=(
            HYBRID_ANSWER_SYSTEM_PROMPT
        ),
    )

    answer = answer.strip()

    if not answer:
        raise RuntimeError(
            "Hybrid answer model returned "
            "an empty answer."
        )

    return answer