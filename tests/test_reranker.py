import pytest
from unittest.mock import Mock

from trustquery.retrieval.reranker import rerank_chunks


def test_rerank_chunks_orders_by_score():
    chunks = [
        {
            "text": "Backups are performed every night.",
            "metadata": {"source": "backup.pdf", "page": 1},
        },
        {
            "text": "Multi-factor authentication is required for administrators.",
            "metadata": {"source": "access.pdf", "page": 2},
        },
    ]

    model = Mock()
    model.predict.return_value = [0.2, 0.9]

    results = rerank_chunks(
        query="Is MFA required for administrators?",
        chunks=chunks,
        model=model,
        top_k=2,
    )

    assert len(results) == 2
    assert results[0]["text"].startswith("Multi-factor authentication")
    assert results[0]["reranker_score"] == pytest.approx(0.9)


def test_rerank_chunks_respects_top_k():
    chunks = [
        {"text": "Chunk one", "metadata": {}},
        {"text": "Chunk two", "metadata": {}},
        {"text": "Chunk three", "metadata": {}},
    ]

    model = Mock()
    model.predict.return_value = [0.1, 0.8, 0.5]

    results = rerank_chunks(
        query="test query",
        chunks=chunks,
        model=model,
        top_k=2,
    )

    assert len(results) == 2
    assert results[0]["text"] == "Chunk two"
    assert results[1]["text"] == "Chunk three"


def test_rerank_chunks_empty_query():
    model = Mock()

    assert rerank_chunks("", [], model) == []


def test_rerank_chunks_empty_chunks():
    model = Mock()

    assert rerank_chunks("authentication", [], model) == []


def test_rerank_chunks_invalid_top_k():
    model = Mock()

    with pytest.raises(ValueError):
        rerank_chunks(
            query="authentication",
            chunks=[{"text": "example", "metadata": {}}],
            model=model,
            top_k=0,
        )