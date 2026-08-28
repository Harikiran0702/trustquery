def retrieval_hit(
    retrieved_chunks: list[dict],
    expected_source: str | None,
    expected_page: int | None,
) -> bool:
    if expected_source is None or expected_page is None:
        return False

    for chunk in retrieved_chunks:
        metadata = chunk.get("metadata", {})

        if (
            metadata.get("source") == expected_source
            and metadata.get("page") == expected_page
        ):
            return True

    return False