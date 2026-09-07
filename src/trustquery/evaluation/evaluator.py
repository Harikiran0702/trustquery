import re

from trustquery.evaluation.metrics import (
    abstention_accuracy,
    abstention_correct,
    retrieval_hit,
    retrieval_hit_rate,
    citation_correctness,
    citation_coverage,
)


def evaluate_retrieval(
    dataset: list[dict],
    retrieve_fn,
) -> dict:
    results = []

    for item in dataset:
        question = item["question"]

        retrieved_chunks = retrieve_fn(question)

        hit = retrieval_hit(
            retrieved_chunks=retrieved_chunks,
            expected_source=item["expected_source"],
            expected_page=item["expected_page"],
        )

        results.append(
            {
                "id": item["id"],
                "question": question,
                "should_answer": item["should_answer"],
                "expected_source": item["expected_source"],
                "expected_page": item["expected_page"],
                "retrieval_hit": hit,
            }
        )

    return {
        "results": results,
        "retrieval_hit_rate": retrieval_hit_rate(results),
    }

def evaluate_generation(
    dataset: list[dict],
    retrieve_fn,
    generate_fn,
) -> dict:
    results = []

    for item in dataset:
        question = item["question"]

        retrieved_chunks = retrieve_fn(question)

        answer = generate_fn(
            question=question,
            retrieved_chunks=retrieved_chunks,
        )

        citations = extract_citations(answer)

        correct = abstention_correct(
            answer=answer,
            should_answer=item["should_answer"],
        )

        coverage = citation_coverage(
            citations=citations,
            should_answer=item["should_answer"],
        )

        correctness = citation_correctness(
            citations=citations,
            expected_source=item["expected_source"],
            expected_page=item["expected_page"],
            should_answer=item["should_answer"],
        )

        results.append(
            {
                "id": item["id"],
                "question": question,
                "should_answer": item["should_answer"],
                "expected_source": item["expected_source"],
                "expected_page": item["expected_page"],
                "answer": answer,
                "citations": citations,
                "abstention_correct": correct,
                "citation_coverage": coverage,
                "citation_correctness": correctness,
            }
        )

    answerable_results = [
        result
        for result in results
        if result["should_answer"]
    ]

    citation_coverage_rate = (
        sum(
            result["citation_coverage"]
            for result in answerable_results
        )
        / len(answerable_results)
        if answerable_results
        else 0.0
    )

    citation_correctness_rate = (
        sum(
            result["citation_correctness"]
            for result in answerable_results
        )
        / len(answerable_results)
        if answerable_results
        else 0.0
    )

    return {
        "results": results,
        "abstention_accuracy": abstention_accuracy(results),
        "citation_coverage": citation_coverage_rate,
        "citation_correctness": citation_correctness_rate,
    }

def extract_citations(answer: str) -> list[dict]:
    """Extract Source/Page citations from a generated answer."""
    matches = re.findall(
        r"\*{0,2}Source:\*{0,2}\s*(.+?),\s*Page:\s*(\d+)",
        answer,
    )

    return [
        {
            "source": source.strip(),
            "page": int(page),
        }
        for source, page in matches
    ]