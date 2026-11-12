"""Explicitly populate the Hugging Face cache for TrustQuery hybrid retrieval."""
from trustquery.hybrid_candidate_sections import EMBEDDER, EMBED_REVISION, RERANKER, RERANK_REVISION


def main():
    from huggingface_hub import snapshot_download

    for repo, revision in ((EMBEDDER, EMBED_REVISION), (RERANKER, RERANK_REVISION)):
        snapshot_download(repo_id=repo, revision=revision, token=False,
                          allow_patterns=['*.json', '*.txt', '*.safetensors', '*.model'])


if __name__ == '__main__':
    main()
