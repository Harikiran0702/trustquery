"""Offline questionnaire validation and blank draft creation (standard library only)."""

import argparse
import csv
from dataclasses import asdict, dataclass
import io
import json
from pathlib import Path
from typing import Literal

from trustquery.file_hashes import text_sha256


class QuestionnaireError(ValueError):
    """Invalid questionnaire file or record."""


@dataclass
class Question:
    id: str
    question: str
    response_type: str
    metadata: dict


@dataclass
class DraftRecord:
    id: str
    question: str
    response_type: str
    metadata: dict
    status: Literal["pending", "needs_evidence"]
    answer: str
    citations: list[dict]
    provenance: dict


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise QuestionnaireError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _constant(value):
    raise QuestionnaireError(f"Invalid JSON constant: {value}")


def _json(text):
    return json.loads(text, object_pairs_hook=_object, parse_constant=_constant)


def _validate(rows: list) -> list[Question]:
    if not isinstance(rows, list) or not rows:
        raise QuestionnaireError("Questionnaire must be a nonempty list of records")
    questions = []
    seen = set()
    required = {"id", "question", "response_type"}
    allowed = required | {"domain", "metadata"}
    for index, row in enumerate(rows, 1):
        prefix = f"Record {index}: "
        if not isinstance(row, dict):
            raise QuestionnaireError(prefix + "expected an object")
        if not required <= row.keys() or row.keys() - allowed:
            raise QuestionnaireError(prefix + "expected id, question, response_type; only domain and metadata are optional")
        for field in required:
            value = row[field]
            if not isinstance(value, str) or not value.strip():
                raise QuestionnaireError(prefix + f"{field} must be a nonempty string")
        identifier = row["id"].strip()
        if identifier in seen:
            raise QuestionnaireError(prefix + f"duplicate id: {identifier}")
        seen.add(identifier)
        if row["response_type"] not in {"yes_no", "short_answer"}:
            raise QuestionnaireError(prefix + "response_type must be yes_no or short_answer")
        metadata = row.get("metadata", {})
        if not isinstance(metadata, dict):
            raise QuestionnaireError(prefix + "metadata must be an object")
        metadata = dict(metadata)
        if "domain" in row:
            domain = row["domain"]
            if not isinstance(domain, str) or not domain.strip():
                raise QuestionnaireError(prefix + "domain must be a nonempty string when supplied")
            if "domain" in metadata:
                raise QuestionnaireError(prefix + "domain supplied twice")
            metadata["domain"] = domain.strip()
        questions.append(Question(identifier, row["question"].strip(), row["response_type"], metadata))
    return questions


def _csv(text: str) -> list[dict]:
    reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
    headers = reader.fieldnames
    if not headers or len(headers) != len(set(headers)):
        raise QuestionnaireError("CSV requires unique column headers")
    required = {"id", "question", "response_type"}
    if not required <= set(headers) or set(headers) - (required | {"domain", "metadata"}):
        raise QuestionnaireError("Invalid CSV headers")
    rows = []
    for row in reader:
        if None in row or any(value is None for value in row.values()):
            raise QuestionnaireError("CSV row has the wrong number of columns")
        if "metadata" in row:
            row["metadata"] = _json(row["metadata"]) if row["metadata"].strip() else {}
        if row.get("domain") == "":
            del row["domain"]
        rows.append(row)
    return rows


def load_questionnaire(path: str | Path) -> tuple[list[Question], dict]:
    """Read only the supplied input; return questions and byte-level provenance.

    IDs are required, trimmed, case-sensitive, and never generated or sorted.
    Unknown top-level fields are rejected rather than propagated as annotations.
    """
    path = Path(path)
    kind = path.suffix.lower().lstrip(".")
    if kind not in {"csv", "json"}:
        raise QuestionnaireError("Input extension must be .csv or .json")
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8-sig")
        rows = _json(text) if kind == "json" else _csv(text)
        questions = _validate(rows)
    except (OSError, UnicodeError, json.JSONDecodeError, csv.Error) as exc:
        raise QuestionnaireError(f"Cannot load {path.name}: {exc}") from exc
    return questions, {
        "source": path.name,
        "input_format": kind,
        "input_sha256": text_sha256(raw),
    }


def create_drafts(path: str | Path) -> list[DraftRecord]:
    """Validate an entire input and create pending drafts without answering it."""
    questions, source = load_questionnaire(path)
    return [
        DraftRecord(
            **asdict(question),
            status="pending",
            answer="",
            citations=[],
            provenance={
                **source,
                "input_record": index,
                "creation_method": "offline_blank_draft",
                "evidence_status": "not_retrieved",
                "review_status": "requires_human_review",
                "operational_proof": False,
            },
        )
        for index, question in enumerate(questions, 1)
    ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    try:
        drafts = create_drafts(args.input)
        # Exclusive creation protects existing inputs, outputs, and references.
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump([asdict(draft) for draft in drafts], stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    except (QuestionnaireError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(f"Created {len(drafts)} pending blank drafts: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
