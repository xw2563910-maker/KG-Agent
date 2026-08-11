from pathlib import Path
from typing import Any

from pypdf import PdfReader


def load_pdf_pages(
    pdf_path: str | Path,
) -> list[dict[str, Any]]:
    """
    Extract text from a PDF page by page.

    Returns one dictionary for each
    non-empty PDF page.
    """
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"PDF path is not a file: {path}"
        )

    if path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Expected a PDF file: {path}"
        )

    reader = PdfReader(
        str(path)
    )

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):
        text = (
            page.extract_text()
            or ""
        ).strip()

        if not text:
            continue

        page_data = {
            "source": path.name,
            "page_number": page_number,
            "text": text,
        }

        pages.append(
            page_data
        )

    if not pages:
        raise RuntimeError(
            "No extractable text found "
            f"in PDF: {path}"
        )

    return pages