from trustquery.retrieval.bm25_retriever import search_bm25
from trustquery.retrieval.retriever import retrieve_chunks
from collections import defaultdict


def reciprocal_rank_fusion(
    dense_results: list[dict],
    sparse_results: list[dict],
    k: int = 60,
) -> list[dict]:
    scores = defaultdict(float)
    result_lookup = {}

    for rank, result in enumerate(dense_results, start=1):
        key = (
            result["text"],
            str(result.get("metadata", {})),
        )

        scores[key] += 1 / (k + rank)
        result_lookup[key] = result

    for rank, result in enumerate(sparse_results, start=1):
        key = (
            result["text"],
            str(result.get("metadata", {})),
        )

        scores[key] += 1 / (k + rank)
        result_lookup[key] = result

    ranked_keys = sorted(
        scores,
        key=scores.get,
        reverse=True,
    )

    fused_results = []

    for key in ranked_keys:
        result = dict(result_lookup[key])
        result["rrf_score"] = scores[key]
        fused_results.append(result)
    return fused_results

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