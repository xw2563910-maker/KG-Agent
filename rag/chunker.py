from typing import Any


DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200


def chunk_pages(
    pages: list[dict[str, Any]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """
    Split page-level PDF text into overlapping chunks
    while preserving source and page metadata.
    """
    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than 0."
        )

    if chunk_overlap < 0:
        raise ValueError(
            "chunk_overlap cannot be negative."
        )

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller "
            "than chunk_size."
        )

    chunks = []

    chunk_index = 1

    for page in pages:
        text = str(
            page.get("text") or ""
        ).strip()

        if not text:
            continue

        start = 0

        while start < len(text):
            end = min(
                start + chunk_size,
                len(text),
            )

            chunk_text = text[
                start:end
            ].strip()

            if chunk_text:
                chunk = {
                    "source": page.get(
                        "source"
                    ),
                    "page_number": page.get(
                        "page_number"
                    ),
                    "chunk_index": chunk_index,
                    "text": chunk_text,
                }

                chunks.append(
                    chunk
                )

                chunk_index += 1

            if end >= len(text):
                break

            start = (
                end - chunk_overlap
            )

    return chunks