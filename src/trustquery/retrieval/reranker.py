from __future__ import annotations
from typing import TYPE_CHECKING
from trustquery.model_settings import RERANKER_REVISION

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder


DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_reranker(
    model_name: str = DEFAULT_RERANKER_MODEL,
    *, revision: str | None = None, local_files_only: bool = True,
) -> CrossEncoder:
    from sentence_transformers import CrossEncoder
    if revision is None:
        if model_name != DEFAULT_RERANKER_MODEL:
            raise ValueError("Custom reranker models require an explicit revision")
        revision = RERANKER_REVISION
    return CrossEncoder(model_name, revision=revision, local_files_only=local_files_only)


def rerank_chunks(
    query: str,
    chunks: list[dict],
    model: CrossEncoder,
    top_k: int = 5,
) -> list[dict]:
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    if not query.strip():
        return []

    if not chunks:
        return []

    pairs = [
        [query, chunk["text"]]
        for chunk in chunks
    ]

    scores = model.predict(pairs)

    ranked = sorted(
        zip(chunks, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    results = []

    for chunk, score in ranked[:top_k]:
        result = dict(chunk)
        result["reranker_score"] = float(score)
        results.append(result)

    return results