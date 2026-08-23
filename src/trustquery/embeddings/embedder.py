from sentence_transformers import SentenceTransformer


_MODEL_NAME = "all-MiniLM-L6-v2"
_model = SentenceTransformer(_MODEL_NAME)


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    embeddings = _model.encode(texts)

    return embeddings.tolist()