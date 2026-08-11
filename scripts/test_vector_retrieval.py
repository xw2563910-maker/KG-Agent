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
from rag.vector_store import (
    VectorStore,
)


def main():
    if len(sys.argv) != 2:
        raise ValueError(
            "Usage: python -m "
            "scripts.test_vector_retrieval "
            "<pdf_path>"
        )

    pdf_path = sys.argv[1]

    query = (
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
        query
    )

    results = vector_store.search(
        query_embedding,
        top_k=5,
    )

    print()
    print(
        f"Query: {query}"
    )

    print(
        f"Indexed chunks: "
        f"{vector_store.index.ntotal}"
    )

    print()

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print(
            f"--- Rank {rank} ---"
        )

        print(
            f"Score: "
            f"{result['score']}"
        )

        print(
            f"Source: "
            f"{result['source']}"
        )

        print(
            f"Page: "
            f"{result['page_number']}"
        )

        print(
            f"Chunk: "
            f"{result['chunk_index']}"
        )

        print()

        print(
            result["text"][:700]
        )

        print()
        print(
            "=" * 60
        )
        print()


if __name__ == "__main__":
    main()