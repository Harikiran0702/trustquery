import json

from trustquery.evaluation.cache import (
    load_generation_cache,
    save_generation_cache,
)


def test_load_generation_cache_missing_file(tmp_path):
    cache_path = tmp_path / "missing.json"

    cache = load_generation_cache(cache_path)

    assert cache == {}


def test_save_and_load_generation_cache(tmp_path):
    cache_path = tmp_path / "generation_cache.json"

    expected = {
        "q001": {
            "answer": "MFA is required.",
        }
    }

    save_generation_cache(
        cache=expected,
        cache_path=cache_path,
    )

    loaded = load_generation_cache(cache_path)

    assert loaded == expected


def test_saved_cache_is_valid_json(tmp_path):
    cache_path = tmp_path / "generation_cache.json"

    save_generation_cache(
        cache={"q001": {"answer": "test"}},
        cache_path=cache_path,
    )

    with cache_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    assert data["q001"]["answer"] == "test"