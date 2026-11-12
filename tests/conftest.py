import os
import pytest


def pytest_configure():
    from trustquery import config  # noqa: F401
    os.environ.pop("GEMINI_API_KEY", None)


@pytest.fixture(autouse=True)
def model_dependencies(request, monkeypatch):
    if request.node.get_closest_marker("model") is None:
        return
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    try:
        pytest.importorskip("torch", exc_type=ImportError)
        pytest.importorskip("sentence_transformers", exc_type=ImportError)
    except OSError as exc:
        pytest.skip(f"Model dependencies unavailable: {exc}")
    name = request.node.name
    try:
        if "hybrid" in name or "live_offline_rank" in name:
            from trustquery.hybrid_candidate_sections import models
            models()
        elif "embed_texts_returns" in name or name == "test_retrieve_chunks":
            from trustquery.embeddings.embedder import load_embedder
            load_embedder()
        elif "q002_cached" in name:
            from trustquery.evaluation.nli import load_nli_backend
            load_nli_backend(local_files_only=True)
    except OSError as exc:
        pytest.skip(f"Local model cache unavailable: {exc}")


@pytest.fixture(autouse=True)
def no_inherited_provider_key(monkeypatch, request):
    from trustquery import config  # noqa: F401
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    if request.node.get_closest_marker('model') is None:
        from trustquery import pipeline
        def unavailable(*args, **kwargs):
            raise OSError('NLI model disabled in unit tests')
        monkeypatch.setattr(pipeline, '_faithfulness', unavailable)
