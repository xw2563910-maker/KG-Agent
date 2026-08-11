import sys

import numpy as np

from rag.chunker import (
    chunk_pages,
)
from rag.embeddings import (
    embed_chunks,
)
from rag.pdf_loader import (
    load_pdf_pages,
)


def main():
    if len(sys.argv) != 2:
        raise ValueError(
            "Usage: python -m "
            "scripts.test_chunk_embeddings "
            "<pdf_path>"
        )

    pdf_path = sys.argv[1]

    pages = load_pdf_pages(
        pdf_path
    )

    chunks = chunk_pages(
        pages
    )

    embeddings = embed_chunks(
        chunks
    )

    print()
    print(
        f"Extracted pages: {len(pages)}"
    )

    print(
        f"Generated chunks: {len(chunks)}"
    )

    print(
        f"Embedding shape: "
        f"{embeddings.shape}"
    )

    print(
        f"Embedding dtype: "
        f"{embeddings.dtype}"
    )

    first_norm = np.linalg.norm(
        embeddings[0]
    )

    print(
        f"First embedding norm: "
        f"{first_norm:.4f}"
    )

    print()

    print(
        "First embedding preview:"
    )

    print(
        embeddings[0][:10]
    )


if __name__ == "__main__":
    main()