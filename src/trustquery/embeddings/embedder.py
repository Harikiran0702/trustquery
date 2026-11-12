from functools import lru_cache
from trustquery.model_settings import EMBEDDING_REVISION

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

@lru_cache(maxsize=1)
def load_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(_MODEL_NAME, revision=EMBEDDING_REVISION,
                               local_files_only=True)

def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    return load_embedder().encode(texts).tolist()
