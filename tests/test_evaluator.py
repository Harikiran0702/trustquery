from trustquery.evaluation.evaluator import evaluate_retrieval


def test_evaluate_retrieval():
    dataset = [
        {
            "id": "q001",
            "question": "Is MFA required?",
            "expected_answer": "Yes.",
            "expected_source": "policy.pdf",
            "expected_page": 1,
            "should_answer": True,
        },
        {
            "id": "q002",
            "question": "Is encryption required?",
            "expected_answer": None,
            "expected_source": None,
            "expected_page": None,
            "should_answer": False,
        },
    ]

    def fake_retrieve(question):
        if question == "Is MFA required?":
            return [
                {
                    "text": "MFA is required.",
                    "metadata": {
                        "source": "policy.pdf",
                        "page": 1,
                    },
                }
            ]

        return []

    report = evaluate_retrieval(dataset, fake_retrieve)

    assert len(report["results"]) == 2
    assert report["results"][0]["retrieval_hit"] is True
    assert report["results"][1]["retrieval_hit"] is False
    assert report["retrieval_hit_rate"] == 1.0