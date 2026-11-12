"""Shared deterministic rank fusion; no model initialization."""
from collections import defaultdict

def _result_identity(result: dict) -> tuple:
    """Return a stable identity shared by dense and sparse result shapes."""
    metadata = result.get("metadata") or {}
    return (
        result.get("text", ""),
        metadata.get("source"),
        metadata.get("page"),
        metadata.get("chunk_index"),
    )


def reciprocal_rank_fusion(
    dense_results: list[dict],
    sparse_results: list[dict],
    k: int = 60,
) -> list[dict]:
    scores = defaultdict(float)
    result_lookup = {}

    for rank, result in enumerate(dense_results, start=1):
        key = _result_identity(result)

        scores[key] += 1 / (k + rank)
        result_lookup[key] = dict(result)

    for rank, result in enumerate(sparse_results, start=1):
        key = _result_identity(result)

        scores[key] += 1 / (k + rank)
        if key in result_lookup:
            merged_result = dict(result_lookup[key])
            merged_result.update(result)
            merged_metadata = dict(result_lookup[key].get("metadata") or {})
            merged_metadata.update(result.get("metadata") or {})
            merged_result["metadata"] = merged_metadata
            result_lookup[key] = merged_result
        else:
            result_lookup[key] = dict(result)

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
