import sys

from rag.pdf_loader import (
    load_pdf_pages,
)


def main():
    if len(sys.argv) != 2:
        raise ValueError(
            "Usage: python -m "
            "scripts.test_pdf_loader "
            "<pdf_path>"
        )

    pdf_path = sys.argv[1]

    pages = load_pdf_pages(
        pdf_path
    )

    print(
        f"Extracted pages: {len(pages)}"
    )

    for page in pages[:3]:
        print()

        print(
            f"--- Page "
            f"{page['page_number']} ---"
        )
        text_preview = page[
            "text"
        ][:500]

        print(
            text_preview
        )


if __name__ == "__main__":
    main()