from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def build_bm25_index(chunks: list[dict]) -> BM25Okapi:
    tokenized_corpus = [tokenize(chunk["text"]) for chunk in chunks]
    return BM25Okapi(tokenized_corpus)

def search_bm25(
    query: str,
    chunks: list[dict],
    top_k: int = 5,
) -> list[dict]:
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    if not query.strip():
        return []

    if not chunks:
        return []

    bm25 = build_bm25_index(chunks)
    tokenized_query = tokenize(query)

    scores = bm25.get_scores(tokenized_query)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:top_k]

    results = []

    for index in ranked_indices:
        chunk = chunks[index]

        result = {
            "text": chunk["text"],
            "metadata": chunk.get("metadata", {}),
            "bm25_score": float(scores[index]),
        }

        results.append(result)

    return results