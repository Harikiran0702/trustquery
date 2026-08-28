from trustquery.evaluation.metrics import retrieval_hit, retrieval_hit_rate

def test_retrieval_hit_returns_true_for_expected_source_and_page():
    chunks = [
        {
            "text": "Some evidence",
            "metadata": {
                "source": "access_control_policy.pdf",
                "page": 2,
            },
        }
    ]

    assert retrieval_hit(
        chunks,
        expected_source="access_control_policy.pdf",
        expected_page=2,
    ) is True


def test_retrieval_hit_returns_false_when_page_does_not_match():
    chunks = [
        {
            "text": "Some evidence",
            "metadata": {
                "source": "access_control_policy.pdf",
                "page": 1,
            },
        }
    ]

    assert retrieval_hit(
        chunks,
        expected_source="access_control_policy.pdf",
        expected_page=2,
    ) is False


def test_retrieval_hit_returns_false_for_empty_results():
    assert retrieval_hit(
        [],
        expected_source="access_control_policy.pdf",
        expected_page=1,
    ) is False


def test_retrieval_hit_returns_false_when_no_expected_evidence():
    chunks = [
        {
            "text": "Some evidence",
            "metadata": {
                "source": "access_control_policy.pdf",
                "page": 1,
            },
        }
    ]

    assert retrieval_hit(
        chunks,
        expected_source=None,
        expected_page=None,
    ) is False

def test_retrieval_hit_rate():
    results = [
        {"should_answer": True, "retrieval_hit": True},
        {"should_answer": True, "retrieval_hit": True},
        {"should_answer": True, "retrieval_hit": True},
        {"should_answer": True, "retrieval_hit": False},
        {"should_answer": False, "retrieval_hit": False},
    ]

    assert retrieval_hit_rate(results) == 0.75


def test_retrieval_hit_rate_all_hits():
    results = [
        {"should_answer": True, "retrieval_hit": True},
        {"should_answer": True, "retrieval_hit": True},
    ]

    assert retrieval_hit_rate(results) == 1.0


def test_retrieval_hit_rate_no_answerable_questions():
    results = [
        {"should_answer": False, "retrieval_hit": False},
    ]

    assert retrieval_hit_rate(results) == 0.0    