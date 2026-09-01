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
def retrieval_hit_rate(results: list[dict]) -> float:
    answerable_results = [
        result for result in results
        if result.get("should_answer") is True
    ]

    if not answerable_results:
        return 0.0

    hits = sum(
        1 for result in answerable_results
        if result.get("retrieval_hit") is True
    )

    return hits / len(answerable_results)

def abstention_correct(
    answer: str,
    should_answer: bool,
) -> bool:
    normalized = answer.strip().lower()

    abstained = (
        "insufficient evidence" in normalized
        or "not enough evidence" in normalized
        or "cannot answer" in normalized
    )

    if should_answer:
        return not abstained

    return abstained


def abstention_accuracy(results: list[dict]) -> float:
    if not results:
        return 0.0

    correct = sum(
        1
        for result in results
        if result.get("abstention_correct") is True
    )

    return correct / len(results)