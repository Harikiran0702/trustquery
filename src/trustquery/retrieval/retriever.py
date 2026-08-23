from trustquery.embeddings.embedder import embed_texts
from trustquery.vectorstore.chroma_store import search_chunks


def retrieve_chunks(collection, query: str, top_k: int = 5):
    if not query.strip():
        raise ValueError("query cannot be empty")

    query_embedding = embed_texts([query])[0]

    return search_chunks(
        collection=collection,
        query_embedding=query_embedding,
        top_k=top_k,
    )