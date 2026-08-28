from trustquery.evaluation.metrics import retrieval_hit, retrieval_hit_rate


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