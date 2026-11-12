import json
from pathlib import Path


def load_generation_cache(cache_path: str | Path) -> dict:
    path = Path(cache_path)

    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_generation_cache(
    cache: dict,
    cache_path: str | Path,
) -> None:
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            cache,
            file,
            indent=2,
            ensure_ascii=False,
        )