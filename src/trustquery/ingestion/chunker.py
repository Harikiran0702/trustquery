from __future__ import annotations

from typing import Any


def chunk_documents(
    documents: list[dict[str, Any]],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[dict[str, Any]]:
    """
    Split document text into overlapping character-based chunks
    while preserving document metadata.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[dict[str, Any]] = []

    for document in documents:
        text = document.get("text", "")

        if not text.strip():
            continue

        metadata = {}

        nested_metadata = document.get("metadata", {})
        if isinstance(nested_metadata, dict):
            metadata.update(nested_metadata)

        for key, value in document.items():
            if key not in {"text", "metadata"}:
                metadata[key] = value

        start = 0
        chunk_index = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]

            chunks.append(
                {
                    "text": chunk_text,
                    "metadata": {
                        **metadata,
                        "chunk_index": chunk_index,
                    },
                }
            )

            if end == len(text):
                break

            start = end - chunk_overlap
            chunk_index += 1

    return chunks