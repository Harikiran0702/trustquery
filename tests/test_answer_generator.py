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

def test_generate_grounded_answer_adds_citation(monkeypatch):
    from trustquery.generation import answer_generator

    retrieved_chunks = [
        {
            "text": "Multi-factor authentication is required for administrative access.",
            "metadata": {
                "source": "access_control_policy.pdf",
                "page": 1,
            },
        }
    ]

    monkeypatch.setattr(
        answer_generator,
        "generate_text",
        lambda prompt: "Yes, multi-factor authentication is required.",
    )

    answer = answer_generator.generate_grounded_answer(
        question="Is MFA required?",
        retrieved_chunks=retrieved_chunks,
    )

    assert "Yes, multi-factor authentication is required." in answer
    assert "Source: access_control_policy.pdf" in answer
    assert "Page: 1" in answer


def test_generate_grounded_answer_preserves_existing_citation(monkeypatch):
    from trustquery.generation import answer_generator

    retrieved_chunks = [
        {
            "text": "Multi-factor authentication is required for administrative access.",
            "metadata": {
                "source": "access_control_policy.pdf",
                "page": 1,
            },
        }
    ]

    monkeypatch.setattr(
        answer_generator,
        "generate_text",
        lambda prompt: (
            "Yes, MFA is required.\n\n"
            "Source: access_control_policy.pdf, Page: 1"
        ),
    )

    answer = answer_generator.generate_grounded_answer(
        question="Is MFA required?",
        retrieved_chunks=retrieved_chunks,
    )

    assert answer.count("Source:") == 1
    assert answer.count("Page:") == 1


def test_generate_grounded_answer_insufficient_evidence(monkeypatch):
    from trustquery.generation import answer_generator

    retrieved_chunks = [
        {
            "text": "User access permissions are reviewed quarterly.",
            "metadata": {
                "source": "access_control_policy.pdf",
                "page": 2,
            },
        }
    ]

    monkeypatch.setattr(
        answer_generator,
        "generate_text",
        lambda prompt: "Insufficient evidence in the provided documents.",
    )

    answer = answer_generator.generate_grounded_answer(
        question="Does the company use AES-256 encryption?",
        retrieved_chunks=retrieved_chunks,
    )

    assert answer == "Insufficient evidence in the provided documents."