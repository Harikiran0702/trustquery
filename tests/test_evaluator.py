from trustquery.evaluation.evaluator import (
    evaluate_generation,
    evaluate_retrieval,
    extract_citations,
)

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

def test_evaluate_generation():
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
        return [{"text": "Evidence", "metadata": {}}]

    def fake_generate(question, retrieved_chunks):
        if question == "Is MFA required?":
            return "Yes, MFA is required."

        return "Insufficient evidence in the provided documents."

    report = evaluate_generation(
        dataset=dataset,
        retrieve_fn=fake_retrieve,
        generate_fn=fake_generate,
    )

    assert len(report["results"]) == 2
    assert report["results"][0]["abstention_correct"] is True
    assert report["results"][1]["abstention_correct"] is True
    assert report["abstention_accuracy"] == 1.0

def test_extract_citations():
    answer = (
        "Yes. MFA is required.\n\n"
        "Source: access_control_policy.pdf, Page: 1"
    )

    citations = extract_citations(answer)

    assert citations == [
        {
            "source": "access_control_policy.pdf",
            "page": 1,
        }
    ]


def test_extract_citations_returns_empty_when_missing():
    answer = "Insufficient evidence in the provided documents."

    assert extract_citations(answer) == []

def test_extract_citations_with_markdown_bold():
    answer = (
        "Yes.\n\n"
        "**Source:** access_control_policy.pdf, Page: 1"
    )

    citations = extract_citations(answer)

    assert citations == [
        {
            "source": "access_control_policy.pdf",
            "page": 1,
        }
    ]    