import sys

from rag.chunker import (
    chunk_pages,
)
from rag.embeddings import (
    embed_chunks,
    embed_query,
)
from rag.pdf_loader import (
    load_pdf_pages,
)
from rag.qa import (
    answer_pdf_question,
)
from rag.vector_store import (
    VectorStore,
)


def main():
    if len(sys.argv) != 2:
        raise ValueError(
            "Usage: python -m "
            "scripts.test_pdf_rag "
            "<pdf_path>"
        )

    pdf_path = sys.argv[1]

    question = (
        "What reinforcement learning "
        "algorithm does EmpRL use?"
    )

    pages = load_pdf_pages(
        pdf_path
    )

    chunks = chunk_pages(
        pages
    )

    chunk_embeddings = embed_chunks(
        chunks
    )

    vector_store = VectorStore(
        chunks,
        chunk_embeddings,
    )

    query_embedding = embed_query(
        question
    )

    retrieved_chunks = vector_store.search(
        query_embedding,
        top_k=5,
    )

    answer = answer_pdf_question(
        question,
        retrieved_chunks,
    )

    print()
    print(
        f"Question: {question}"
    )

    print()
    print("Answer:")
    print(answer)

    print()
    print("Retrieved evidence:")

    for index, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):
        print(
            f"[Evidence {index}] "
            f"{chunk['source']} | "
            f"Page {chunk['page_number']} | "
            f"Chunk {chunk['chunk_index']} | "
            f"Score {chunk['score']}"
        )


if __name__ == "__main__":
    main()