from trustquery.vectorstore.chroma_store import (
    add_chunks,
    get_collection,
)


def test_get_collection(tmp_path):
    collection = get_collection(
        collection_name="test_collection",
        persist_directory=str(tmp_path),
    )

    assert collection.name == "test_collection"



def test_add_chunks(tmp_path):
    collection = get_collection(
        collection_name="test_add_chunks",
        persist_directory=str(tmp_path),
    )

    chunks = [
        {
            "text": "Multi-factor authentication is required.",
            "source": "access_control.pdf",
            "page": 1,
        },
        {
            "text": "Passwords must be rotated according to policy.",
            "source": "access_control.pdf",
            "page": 2,
        },
    ]

    embeddings = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    add_chunks(collection, chunks, embeddings)

    assert collection.count() == 2


def test_add_chunks_mismatched_lengths(tmp_path):
    collection = get_collection(
        collection_name="test_mismatch",
        persist_directory=str(tmp_path),
    )

    chunks = [{"text": "Example text", "source": "test.pdf"}]
    embeddings = []

    try:
        add_chunks(collection, chunks, embeddings)
        assert False
    except ValueError:
        assert True