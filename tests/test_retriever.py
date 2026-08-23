from trustquery.embeddings.embedder import embed_texts
from trustquery.retrieval.retriever import retrieve_chunks
from trustquery.vectorstore.chroma_store import (
    add_chunks,
    get_collection,
)


def test_retrieve_chunks(tmp_path):
    collection = get_collection(
        collection_name="test_retriever",
        persist_directory=str(tmp_path),
    )

    chunks = [
        {
            "text": "Multi-factor authentication is required for privileged accounts.",
            "source": "access_control.pdf",
            "page": 1,
        },
        {
            "text": "Backups are performed daily and retained for thirty days.",
            "source": "backup_policy.pdf",
            "page": 2,
        },
    ]

    embeddings = embed_texts(
        [chunk["text"] for chunk in chunks]
    )

    add_chunks(
        collection=collection,
        chunks=chunks,
        embeddings=embeddings,
    )

    results = retrieve_chunks(
        collection=collection,
        query="Is MFA required for privileged users?",
        top_k=1,
    )

    assert len(results) == 1
    assert "Multi-factor authentication" in results[0]["text"]
    assert results[0]["metadata"]["source"] == "access_control.pdf"

def test_retrieve_chunks_empty_query(tmp_path):
    collection = get_collection(
        collection_name="test_empty_query",
        persist_directory=str(tmp_path),
    )

    try:
        retrieve_chunks(
            collection=collection,
            query="   ",
        )
        assert False
    except ValueError:
        assert True