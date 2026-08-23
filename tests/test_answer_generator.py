import pytest

from trustquery.generation.answer_generator import (
    build_grounded_prompt,
    format_evidence,
)


def test_format_evidence():
    chunks = [
        {
            "text": "Multi-factor authentication is required.",
            "metadata": {
                "source": "access_control.pdf",
                "page": 2,
            },
        }
    ]

    evidence = format_evidence(chunks)

    assert "[Evidence 1]" in evidence
    assert "access_control.pdf" in evidence
    assert "Page: 2" in evidence
    assert "Multi-factor authentication is required." in evidence


def test_build_grounded_prompt():
    chunks = [
        {
            "text": "Multi-factor authentication is required.",
            "metadata": {
                "source": "access_control.pdf",
                "page": 2,
            },
        }
    ]

    prompt = build_grounded_prompt(
        question="Does the company require MFA?",
        retrieved_chunks=chunks,
    )

    assert "Does the company require MFA?" in prompt
    assert "access_control.pdf" in prompt
    assert "Multi-factor authentication is required." in prompt


def test_build_grounded_prompt_empty_question():
    with pytest.raises(ValueError):
        build_grounded_prompt(
            question="   ",
            retrieved_chunks=[{"text": "example", "metadata": {}}],
        )


def test_build_grounded_prompt_empty_chunks():
    with pytest.raises(ValueError):
        build_grounded_prompt(
            question="Does the company require MFA?",
            retrieved_chunks=[],
        )