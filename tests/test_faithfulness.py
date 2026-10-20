import pytest

from trustquery.evaluation.faithfulness import extract_claims, faithfulness_score


def test_extract_claims_splits_basic_sentences():
    assert extract_claims("  Access is restricted. Is it audited? Yes!  ") == [
        "Access is restricted.", "Is it audited?", "Yes!"
    ]


def test_extract_claims_removes_standalone_citation_lines():
    answer = (
        "Access is restricted.\n"
        "Source: access_control_policy.pdf, Page: 1\n"
        "Changes are audited.\n"
        "  Source: audit_policy.pdf\n"
        "  Page: 2\n"
        "Source: labels identify evidence."
    )
    assert extract_claims(answer) == [
        "Access is restricted.", "Changes are audited."
    ]
    assert extract_claims("The Source: field identifies evidence.") == [
        "The Source: field identifies evidence."
    ]


def test_extract_claims_returns_empty_for_blank_or_citation_only_answers():
    for answer in ("", " \n\t ", "Source: policy.pdf\nPage: 1"):
        assert extract_claims(answer) == []


def test_faithfulness_score_returns_one_when_all_claims_are_supported():
    assert faithfulness_score(["First.", "Second."], [True, True]) == 1.0


def test_faithfulness_score_returns_proportion_supported():
    assert faithfulness_score(
        ["First.", "Second.", "Third."], [True, False, True]
    ) == pytest.approx(2 / 3)
    assert faithfulness_score(["First."], [False]) == 0.0


def test_faithfulness_score_returns_zero_for_empty_claims():
    assert faithfulness_score([], []) == 0.0


def test_faithfulness_score_rejects_unequal_lengths():
    for claims, supported in ((["First."], []), ([], [True])):
        with pytest.raises(ValueError, match="equal lengths"):
            faithfulness_score(claims, supported)
