import re
from dataclasses import asdict

from trustquery.evaluation.faithfulness import answer_faithfulness
from trustquery.evaluation.nli import NLIBackend

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
    *,
    nli_backend: NLIBackend | None = None,
) -> dict:
    """Evaluate generation, optionally verifying claims with an injected backend.

    No backend is loaded here. With scoring disabled, scores are None. When
    enabled, the aggregate is the mean over ALL answerable cases, including
    zero for empty/abstaining/no-claim/missing-evidence answers. Non-answerable
    cases still receive details but are excluded from that mean. No answerable
    cases yields None. Counts expose how many cases received NLI verdicts.
    Abstention uses the existing metric's phrase heuristic. NLI/validation
    failures propagate rather than returning a partial or successful report.
    """
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

        faithfulness = {
            "faithfulness_score": None,
            "faithfulness_status": "disabled",
            "faithfulness_claims": [],
        }
        if nli_backend is not None:
            # Reuse the existing abstention definition without changing metrics.
            abstained = abstention_correct(answer, should_answer=False)
            if not answer.strip():
                status = "empty_answer"
            elif abstained:
                status = "abstained"
            else:
                status = "scored"
            if status == "scored":
                verification = answer_faithfulness(
                    answer, retrieved_chunks, nli_backend,
                )
                faithfulness["faithfulness_claims"] = [
                    asdict(claim) for claim in verification.claims
                ]
                score = verification.score
                if not verification.claims:
                    status = "no_claims"
                elif not retrieved_chunks:
                    status = "missing_evidence"
            else:
                score = 0.0
            faithfulness.update(
                faithfulness_score=score, faithfulness_status=status,
            )

        results.append(
            {
                "id": item["id"],
                "question": question,
                "should_answer": item["should_answer"],
                "expected_source": item["expected_source"],
                "expected_page": item["expected_page"],
                "answer": answer,
                **faithfulness,
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
        "faithfulness_score": (
            sum(result["faithfulness_score"] for result in answerable_results)
            / len(answerable_results)
            if nli_backend is not None and answerable_results else None
        ),
        "faithfulness_answerable_count": len(answerable_results),
        "faithfulness_scored_count": sum(
            result["faithfulness_status"] == "scored"
            for result in answerable_results
        ),
        "faithfulness_unscored_count": sum(
            result["faithfulness_status"] != "scored"
            for result in answerable_results
        ),
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