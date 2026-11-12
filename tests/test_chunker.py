import pytest

from trustquery.ingestion.chunker import chunk_documents


def test_chunk_documents_preserves_metadata():
    documents = [
        {
            "text": "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            "metadata": {
                "source": "policy.pdf",
                "page": 1,
            },
        }
    ]

    chunks = chunk_documents(
        documents,
        chunk_size=10,
        chunk_overlap=2,
    )

    assert len(chunks) > 1
    assert chunks[0]["metadata"]["source"] == "policy.pdf"
    assert chunks[0]["metadata"]["page"] == 1
    assert chunks[0]["metadata"]["chunk_index"] == 0
    assert chunks[1]["metadata"]["chunk_index"] == 1


def test_chunk_documents_has_overlap():
    documents = [
        {
            "text": "abcdefghijklmnop",
            "metadata": {},
        }
    ]

    chunks = chunk_documents(
        documents,
        chunk_size=10,
        chunk_overlap=2,
    )

    assert chunks[0]["text"] == "abcdefghij"
    assert chunks[1]["text"].startswith("ij")


def test_empty_documents_are_skipped():
    documents = [
        {
            "text": "",
            "metadata": {"source": "empty.txt"},
        }
    ]

    chunks = chunk_documents(documents)

    assert chunks == []


def test_invalid_chunk_size_raises_error():
    with pytest.raises(ValueError):
        chunk_documents([], chunk_size=0)


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_documents(
            [],
            chunk_size=100,
            chunk_overlap=100,
        )