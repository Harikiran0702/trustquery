from trustquery.retrieval.bm25_retriever import search_bm25
from trustquery.retrieval.retriever import retrieve_chunks
from trustquery.retrieval.reranker import rerank_chunks
from trustquery.retrieval.rrf import reciprocal_rank_fusion


def hybrid_retrieve(
    query: str,
    collection,
    chunks: list[dict],
    top_k: int = 5,
    candidate_k: int = 10,
) -> list[dict]:
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    if candidate_k <= 0:
        raise ValueError("candidate_k must be greater than 0")

    if not query.strip():
        return []

    dense_results = retrieve_chunks(
        query=query,
        collection=collection,
        top_k=candidate_k,
    )

    sparse_results = search_bm25(
        query=query,
        chunks=chunks,
        top_k=candidate_k,
    )

    fused_results = reciprocal_rank_fusion(
        dense_results=dense_results,
        sparse_results=sparse_results,
    )

    return fused_results[:top_k]

def retrieve_and_rerank(
    query: str,
    collection,
    chunks: list[dict],
    reranker_model,
    top_k: int = 5,
    candidate_k: int = 10,
) -> list[dict]:
    candidates = hybrid_retrieve(
        query=query,
        collection=collection,
        chunks=chunks,
        top_k=candidate_k,
        candidate_k=candidate_k,
    )

    return rerank_chunks(
        query=query,
        chunks=candidates,
        model=reranker_model,
        top_k=top_k,
    )
