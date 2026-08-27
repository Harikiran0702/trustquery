from unittest.mock import patch

from trustquery.retrieval.hybrid_retriever import (
    hybrid_retrieve,
    reciprocal_rank_fusion,
    retrieve_and_rerank,
)


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


def test_rrf_deduplicates_same_chunk_with_different_result_metadata():
    dense_results = [
        {
            "text": "Multi-factor authentication is required.",
            "metadata": {
                "source": "access.pdf",
                "page": 1,
                "chunk_index": 0,
                "chroma_id": "chunk-0",
            },
            "distance": 0.12,
        }
    ]
    sparse_results = [
        {
            "text": "Multi-factor authentication is required.",
            "metadata": {
                "source": "access.pdf",
                "page": 1,
                "chunk_index": 0,
            },
            "bm25_score": 3.5,
        }
    ]

    results = reciprocal_rank_fusion(dense_results, sparse_results)

    assert len(results) == 1
    assert results[0]["rrf_score"] == (1 / 61) + (1 / 61)
    assert results[0]["distance"] == 0.12
    assert results[0]["bm25_score"] == 3.5
    assert results[0]["metadata"]["chroma_id"] == "chunk-0"


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

@patch("trustquery.retrieval.hybrid_retriever.rerank_chunks")
@patch("trustquery.retrieval.hybrid_retriever.hybrid_retrieve")
def test_retrieve_and_rerank(
    mock_hybrid_retrieve,
    mock_rerank_chunks,
):
    mock_hybrid_retrieve.return_value = [
        {
            "text": "Multi-factor authentication is required.",
            "metadata": {"source": "access.pdf", "page": 1},
            "rrf_score": 0.03,
        }
    ]

    mock_rerank_chunks.return_value = [
        {
            "text": "Multi-factor authentication is required.",
            "metadata": {"source": "access.pdf", "page": 1},
            "rrf_score": 0.03,
            "reranker_score": 0.9,
        }
    ]

    model = object()

    results = retrieve_and_rerank(
        query="Is MFA required?",
        collection=object(),
        chunks=[],
        reranker_model=model,
        top_k=3,
        candidate_k=10,
    )

    assert len(results) == 1
    assert results[0]["reranker_score"] == 0.9

    mock_hybrid_retrieve.assert_called_once()
    mock_rerank_chunks.assert_called_once()


@patch("trustquery.retrieval.hybrid_retriever.rerank_chunks")
@patch("trustquery.retrieval.hybrid_retriever.search_bm25")
@patch("trustquery.retrieval.hybrid_retriever.retrieve_chunks")
def test_retrieve_and_rerank_deduplicates_retriever_results(
    mock_retrieve_chunks,
    mock_search_bm25,
    mock_rerank_chunks,
):
    dense_result = {
        "text": "Multi-factor authentication is required.",
        "metadata": {
            "source": "access.pdf",
            "page": 1,
            "chunk_index": 0,
            "chroma_id": "chunk-0",
        },
        "distance": 0.12,
    }
    sparse_result = {
        "text": dense_result["text"],
        "metadata": {
            "source": "access.pdf",
            "page": 1,
            "chunk_index": 0,
        },
        "bm25_score": 3.5,
    }
    mock_retrieve_chunks.return_value = [dense_result]
    mock_search_bm25.return_value = [sparse_result]
    mock_rerank_chunks.side_effect = lambda **kwargs: kwargs["chunks"]

    results = retrieve_and_rerank(
        query="Is MFA required?",
        collection=object(),
        chunks=[sparse_result],
        reranker_model=object(),
        top_k=2,
        candidate_k=10,
    )

    assert len(results) == 1
    assert results[0]["rrf_score"] == (1 / 61) + (1 / 61)
    assert results[0]["distance"] == 0.12
    assert results[0]["bm25_score"] == 3.5
