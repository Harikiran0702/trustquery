"""Opt-in startup for the two pinned search models."""
import logging

import streamlit as st

from trustquery.hybrid_candidate_sections import (
    EMBEDDER, EMBED_REVISION, RERANKER, RERANK_REVISION, models,
)


@st.cache_resource(show_spinner=False)
def download_search_models():
    try:
        from huggingface_hub import snapshot_download

        for repo, revision in ((EMBEDDER, EMBED_REVISION), (RERANKER, RERANK_REVISION)):
            snapshot_download(repo_id=repo, revision=revision, token=False,
                              allow_patterns=['*.json', '*.txt', '*.safetensors', '*.model'])
        models()
    except Exception as exc:
        logging.getLogger(__name__).warning('Search model startup failed (%s)', type(exc).__name__)
        return False
    return True
