import re
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
    *, normalize_query: bool = False,
) -> list[dict]:
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    if not query.strip():
        return []

    if not chunks:
        return []

    if normalize_query:
        bm25 = BM25Okapi([normalized_tokens(chunk['text']) for chunk in chunks])
        tokenized_query = normalized_tokens(query)
    else:
        bm25 = build_bm25_index(chunks)
        tokenized_query = tokenize(query)

    scores = bm25.get_scores(tokenized_query)
    if normalize_query:
        # A policy's document title supplies useful topic context to every chunk.
        for i, chunk in enumerate(chunks):
            title = chunk.get('metadata', {}).get('source', '').replace('_', ' ')
            scores[i] += 5 * sum(bm25.idf.get(t, 0) for t in set(tokenized_query) & query_terms(title))

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

STOP = set('a an the your our do does is are have has how what which who when where why often written policy policies require requires and or of on in at by with for to you'.split())


def normalized_tokens(text):
    terms = []
    for token in re.findall(r'[a-z0-9]+', text.lower()):
        if token in STOP and token not in {'require', 'requires'}:
            continue
        term = {'required': 'require', 'requires': 'require', 'reviews': 'review'}.get(token, token)
        if term.endswith('ies'):
            term = term[:-3] + 'y'
        elif term.endswith('s') and not term.endswith(('ss', 'us')):
            term = term[:-1]
        terms.append(term)
    return terms


def query_terms(text):
    return set(normalized_tokens(text))
