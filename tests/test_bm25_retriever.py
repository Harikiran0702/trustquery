import pytest

from trustquery.retrieval.bm25_retriever import search_bm25


def test_search_bm25_returns_relevant_chunk():
    chunks = [
        {
            "text": "Multi-factor authentication is required for administrative access.",
            "metadata": {"source": "access_control_policy.pdf", "page": 1},
        },
        {
            "text": "Backups are performed every night and retained for thirty days.",
            "metadata": {"source": "backup_policy.pdf", "page": 2},
        },
    ]

    results = search_bm25(
        "Is multi-factor authentication required?",
        chunks,
        top_k=1,
    )

    assert len(results) == 1
    assert "Multi-factor authentication" in results[0]["text"]
    assert results[0]["metadata"]["page"] == 1
    assert "bm25_score" in results[0]


def test_search_bm25_empty_query():
    chunks = [{"text": "Example text", "metadata": {}}]

    assert search_bm25("", chunks) == []


def test_search_bm25_empty_chunks():
    assert search_bm25("authentication", []) == []


def test_search_bm25_invalid_top_k():
    chunks = [{"text": "Example text", "metadata": {}}]

    with pytest.raises(ValueError):
        search_bm25("example", chunks, top_k=0)