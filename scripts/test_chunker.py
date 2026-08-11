import sys

from rag.pdf_loader import (
    load_pdf_pages,
)
from rag.chunker import (
    chunk_pages,
)


def main():
    if len(sys.argv) != 2:
        raise ValueError(
            "Usage: python -m "
            "scripts.test_chunker "
            "<pdf_path>"
        )

    pdf_path = sys.argv[1]

    pages = load_pdf_pages(
        pdf_path
    )

    chunks = chunk_pages(
        pages
    )

    print(
        f"Extracted pages: {len(pages)}"
    )

    print(
        f"Generated chunks: {len(chunks)}"
    )

    print()

    for chunk in chunks[:5]:
        print(
            f"--- Chunk "
            f"{chunk['chunk_index']} ---"
        )

        print(
            f"Source: "
            f"{chunk['source']}"
        )

        print(
            f"Page: "
            f"{chunk['page_number']}"
        )

        print(
            f"Length: "
            f"{len(chunk['text'])}"
        )

        print()

        print(
            chunk["text"][:500]
        )

        print()
        print(
            "=" * 60
        )
        print()


if __name__ == "__main__":
    main()