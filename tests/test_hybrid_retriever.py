from trustquery.retrieval.hybrid_retriever import reciprocal_rank_fusion


def test_rrf_combines_dense_and_sparse_results():
    dense_results = [
        {
            "text": "Multi-factor authentication is required.",
            "metadata": {"source": "access.pdf", "page": 1},
        },
        {
            "text": "Passwords must be twelve characters.",
            "metadata": {"source": "access.pdf", "page": 2},
        },
    ]

    sparse_results = [
        {
            "text": "Multi-factor authentication is required.",
            "metadata": {"source": "access.pdf", "page": 1},
            "bm25_score": 3.5,
        },
        {
            "text": "Administrative accounts are reviewed quarterly.",
            "metadata": {"source": "access.pdf", "page": 3},
            "bm25_score": 2.1,
        },
    ]

    results = reciprocal_rank_fusion(
        dense_results,
        sparse_results,
    )

    assert len(results) == 3
    assert results[0]["text"] == "Multi-factor authentication is required."
    assert "rrf_score" in results[0]


def test_rrf_handles_empty_dense_results():
    sparse_results = [
        {
            "text": "Example text",
            "metadata": {"source": "example.pdf"},
        }
    ]

    results = reciprocal_rank_fusion([], sparse_results)

    assert len(results) == 1
    assert results[0]["text"] == "Example text"


def test_rrf_handles_empty_sparse_results():
    dense_results = [
        {
            "text": "Example text",
            "metadata": {"source": "example.pdf"},
        }
    ]

    results = reciprocal_rank_fusion(dense_results, [])

    assert len(results) == 1
    assert results[0]["text"] == "Example text"


def test_rrf_empty_results():
    assert reciprocal_rank_fusion([], []) == []

import pytest
from unittest.mock import patch

from trustquery.retrieval.hybrid_retriever import (
    hybrid_retrieve,
    reciprocal_rank_fusion,
)


@patch("trustquery.retrieval.hybrid_retriever.search_bm25")
@patch("trustquery.retrieval.hybrid_retriever.retrieve_chunks")
def test_hybrid_retrieve_combines_retrievers(
    mock_retrieve_chunks,
    mock_search_bm25,
):
    mock_retrieve_chunks.return_value = [
        {
            "text": "Multi-factor authentication is required.",
            "metadata": {"source": "access.pdf", "page": 1},
        }
    ]

    mock_search_bm25.return_value = [
        {
            "text": "Multi-factor authentication is required.",
            "metadata": {"source": "access.pdf", "page": 1},
            "bm25_score": 3.5,
        },
        {
            "text": "Administrative access is reviewed.",
            "metadata": {"source": "access.pdf", "page": 2},
            "bm25_score": 2.0,
        },
    ]

    results = hybrid_retrieve(
        query="Is MFA required?",
        collection=object(),
        chunks=[],
        top_k=2,
        candidate_k=5,
    )

    assert len(results) == 2
    assert results[0]["text"] == "Multi-factor authentication is required."
    assert "rrf_score" in results[0]


def test_hybrid_retrieve_empty_query():
    results = hybrid_retrieve(
        query="",
        collection=object(),
        chunks=[],
    )

    assert results == []


def test_hybrid_retrieve_invalid_top_k():
    with pytest.raises(ValueError):
        hybrid_retrieve(
            query="authentication",
            collection=object(),
            chunks=[],
            top_k=0,
        )


def test_hybrid_retrieve_invalid_candidate_k():
    with pytest.raises(ValueError):
        hybrid_retrieve(
            query="authentication",
            collection=object(),
            chunks=[],
            candidate_k=0,
        )