
import pytest

from trustquery.evaluation.faithfulness import (
    extract_claims,
    faithfulness_score,
)


def test_extract_claims_splits_basic_sentences():
    assert extract_claims(
        "  Access is restricted. Is it audited? Yes!  "
    ) == ["Access is restricted."]


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
        "Access is restricted.",
        "Changes are audited.",
    ]

    assert extract_claims(
        "The Source: field identifies evidence."
    ) == ["The Source: field identifies evidence."]


def test_extract_claims_returns_empty_for_blank_or_citation_only_answers():
    for answer in (
        "",
        " \n\t ",
        "Source: policy.pdf\nPage: 1",
    ):
        assert extract_claims(answer) == []


@pytest.mark.parametrize("label", [
    "Source:",
    "Page:",
    "source:",
    "  PAGE :",
    "**Source:**",
    "**Page:**",
    "**Source**:",
    "**Page**:",
    "__Source:__",
    "__Page:__",
    "*Source:*",
    "_Page:_",
])
def test_extract_claims_removes_plain_and_markdown_citation_labels(label):
    answer = (
        f"Access is restricted.\n"
        f"{label} policy.pdf, Page: 1\n"
        f"Changes are audited."
    )

    assert extract_claims(answer) == [
        "Access is restricted.",
        "Changes are audited.",
    ]

    assert extract_claims(f"{label} policy.pdf, Page: 1") == []


@pytest.mark.parametrize("answer", [
    "Is access restricted?",
    "Is access restricted? Is it audited?",
    "Yes!",
    "YES!",
    "yes",
    "Yes.",
    "Yes!!!",
    "**Yes!**",
    "**Is access restricted?**",
    "Is access restricted? Yes!",
])
def test_extract_claims_excludes_questions_and_standalone_yes(answer):
    assert extract_claims(answer) == []


@pytest.mark.parametrize("statement", [
    "Yes, access is restricted.",
    "Yes! Access is restricted.",
    "Access is restricted!",
    "Access is restricted",
    "The **Source:** field identifies evidence.",
    "Source control is enabled.",
    "Page access is logged.",
    "**Access is restricted.**",
])
def test_extract_claims_preserves_factual_statements(statement):
    expected = statement.removeprefix("Yes! ")

    assert extract_claims(statement) == [expected]


def test_extract_claims_handles_mixed_answer():
    answer = (
        "Is access restricted? Yes! Access is restricted.\n"
        "**Source:** access_control_policy.pdf, Page: 1\n"
        "Is it audited?\n"
        "Changes are audited!\n"
        "Page: 2\n"
        "Logs are retained"
    )

    assert extract_claims(answer) == [
        "Access is restricted.",
        "Changes are audited!",
        "Logs are retained",
    ]


@pytest.mark.parametrize("reply", [
    "No",
    "N/A",
    "NA",
    "Unknown",
])
@pytest.mark.parametrize("decoration", [
    "{}",
    "{}.",
    "{}!",
    "{}...",
    "{}?",
    "{}:",
    "{};",
    "{},",
    "**{}**",
    "__{}__",
    "*{}*",
    "_{}_",
    "**{}!**",
    "**{}**.",
])
def test_extract_claims_excludes_standalone_non_claim_replies(
    reply,
    decoration,
):
    assert extract_claims(decoration.format(reply.lower())) == []
    assert extract_claims(decoration.format(reply.upper())) == []


@pytest.mark.parametrize("statement", [
    "MFA is not required.",
    "No, MFA is not required.",
    "No access is permitted.",
    "N/A is the configured default.",
    "NA is a region code.",
    "Unknown users are denied access.",
    "**MFA is not required.**",
])
def test_extract_claims_preserves_negative_and_reply_prefixed_facts(
    statement,
):
    assert extract_claims(statement) == [statement]


def test_extract_claims_filters_non_claim_replies_in_mixed_answer():
    answer = (
        "No. MFA is not required. N/A! NA. "
        "Unknown. Access is logged."
    )

    assert extract_claims(answer) == [
        "MFA is not required.",
        "Access is logged.",
    ]


def test_faithfulness_score_returns_one_when_all_claims_are_supported():
    assert faithfulness_score(
        ["First.", "Second."],
        [True, True],
    ) == 1.0


def test_faithfulness_score_returns_proportion_supported():
    assert faithfulness_score(
        ["First.", "Second.", "Third."],
        [True, False, True],
    ) == pytest.approx(2 / 3)

    assert faithfulness_score(
        ["First."],
        [False],
    ) == 0.0


def test_faithfulness_score_returns_zero_for_empty_claims():
    assert faithfulness_score([], []) == 0.0


def test_faithfulness_score_rejects_unequal_lengths():
    for claims, supported in (
        (["First."], []),
        ([], [True]),
    ):
        with pytest.raises(ValueError, match="equal lengths"):
            faithfulness_score(claims, supported)
