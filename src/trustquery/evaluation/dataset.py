import json
from pathlib import Path


def load_golden_dataset(path: str) -> list[dict]:
    dataset_path = Path(path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Golden dataset not found: {path}")

    with dataset_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("Golden dataset must be a list of test cases.")

    required_fields = {
        "id",
        "question",
        "expected_answer",
        "expected_source",
        "expected_page",
        "should_answer",
    }

    for item in data:
        missing = required_fields - item.keys()
        if missing:
            raise ValueError(
                f"Test case {item.get('id', '<unknown>')} "
                f"is missing fields: {sorted(missing)}"
            )

    return data