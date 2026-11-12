import json

import pytest

from trustquery.evaluation.dataset import load_golden_dataset


def test_load_golden_dataset(tmp_path):
    dataset = [
        {
            "id": "q001",
            "question": "Is MFA required?",
            "expected_answer": "Yes.",
            "expected_source": "policy.pdf",
            "expected_page": 1,
            "should_answer": True,
        }
    ]

    path = tmp_path / "golden.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")

    result = load_golden_dataset(str(path))

    assert len(result) == 1
    assert result[0]["id"] == "q001"
    assert result[0]["should_answer"] is True


def test_missing_dataset_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_golden_dataset("does_not_exist.json")


def test_dataset_must_be_list(tmp_path):
    path = tmp_path / "golden.json"
    path.write_text(json.dumps({"id": "q001"}), encoding="utf-8")

    with pytest.raises(ValueError, match="must be a list"):
        load_golden_dataset(str(path))


def test_missing_required_field_raises_error(tmp_path):
    dataset = [
        {
            "id": "q001",
            "question": "Is MFA required?",
        }
    ]

    path = tmp_path / "golden.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")

    with pytest.raises(ValueError, match="missing fields"):
        load_golden_dataset(str(path))