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

def citation_coverage(
    citations: list[dict],
    should_answer: bool,
) -> float:
    """Check whether an answer contains citations when citations are expected.

    Args:
        citations: Citations produced with the generated answer.
        should_answer: Whether the golden dataset expects the question
            to be answerable.

    Returns:
        1.0 when citation behavior is correct, otherwise 0.0.
    """
    if should_answer:
        return 1.0 if citations else 0.0

    # For unanswerable questions, no citation should be produced.
    return 1.0 if not citations else 0.0


def citation_correctness(
    citations: list[dict],
    expected_source: str | None,
    expected_page: int | None,
    should_answer: bool,
) -> float:
    """Check whether at least one citation matches expected evidence.

    Args:
        citations: Citations produced with the generated answer.
        expected_source: Expected source document.
        expected_page: Expected source page.
        should_answer: Whether the question should be answered.

    Returns:
        1.0 if citation behavior is correct, otherwise 0.0.
    """
    if not should_answer:
        return 1.0 if not citations else 0.0

    if not citations:
        return 0.0

    for citation in citations:
        source = citation.get("source")
        page = citation.get("page")

        source_matches = (
            expected_source is None or source == expected_source
        )
        page_matches = (
            expected_page is None or page == expected_page
        )

        if source_matches and page_matches:
            return 1.0

    return 0.0