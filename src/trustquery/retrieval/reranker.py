from sentence_transformers import CrossEncoder


DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_reranker(
    model_name: str = DEFAULT_RERANKER_MODEL,
) -> CrossEncoder:
    return CrossEncoder(model_name)


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