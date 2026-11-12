"""Offline input contracts and draft safety; no models or API clients needed."""

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

import pytest

from trustquery.file_hashes import text_sha256
from trustquery.batch import QuestionnaireError, create_drafts, load_questionnaire, main


def row(**changes):
    return {"id": "q2", "question": "Is MFA required?", "response_type": "yes_no", **changes}


@pytest.mark.parametrize("kind", ["csv", "json"])
def test_valid_order_metadata_and_determinism(tmp_path, kind):
    path = tmp_path / f"questions.{kind}"
    if kind == "json":
        path.write_text(json.dumps([row(domain="access", metadata={"owner": "team"}), row(id="q1")]), encoding="utf-8-sig")
    else:
        path.write_text('id,question,response_type,domain,metadata\nq2,"Is MFA required, including admins?",yes_no,access,"{""owner"":""team""}"\nq1,Describe access.,short_answer,,\n', encoding="utf-8-sig")
    questions, provenance = load_questionnaire(path)
    assert [q.id for q in questions] == ["q2", "q1"]
    assert questions[0].metadata == {"owner": "team", "domain": "access"}
    assert questions[1].metadata == {}
    assert provenance["input_sha256"] == text_sha256(path.read_bytes())
    drafts = create_drafts(path)
    assert drafts == create_drafts(path)
    for index, draft in enumerate(drafts, 1):
        assert draft.status == "pending"
        assert draft.answer == "" and draft.citations == []
        assert draft.provenance == {**provenance, "input_record": index,
            "creation_method": "offline_blank_draft", "evidence_status": "not_retrieved",
            "review_status": "requires_human_review", "operational_proof": False}
    drafts[0].citations.append({"source": "test"})
    assert drafts[1].citations == []


@pytest.mark.parametrize("records", [
    [], {}, [None], [row(), row()], [row(), row(id=" q2 ")],
    [row(id="")], [row(id=1)], [row(question="  ")], [row(question=None)],
    [{"id": "q1", "response_type": "yes_no"}],
    [row(response_type="boolean")], [row(response_type=True)],
    [row(metadata=[])], [row(domain=42)], [row(domain=" ")],
    [row(domain="access", metadata={"domain": "other"})],
    [row(answer="yes")], [row(expected_answer="yes")], [row(citations=[])],
])
def test_invalid_json_records(tmp_path, records):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    with pytest.raises(QuestionnaireError):
        create_drafts(path)


@pytest.mark.parametrize("text", [
    "", "{", '[{"id":"a","id":"b"}]',
    '[{"id":"a","question":"Q?","response_type":"yes_no","metadata":{"x":NaN}}]',
])
def test_malformed_json(tmp_path, text):
    path = tmp_path / "bad.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(QuestionnaireError):
        create_drafts(path)


@pytest.mark.parametrize("text", [
    "", "id,question\na,Q?\n", "id,id,question,response_type\na,b,Q?,yes_no\n",
    "id,question,response_type,answer\na,Q?,yes_no,yes\n",
    "id,question,response_type\na,Q?\n", "id,question,response_type\na,Q?,yes_no,extra\n",
    'id,question,response_type\na,"Unclosed,yes_no\n',
    "id,question,response_type\na,  ,yes_no\n",
    "id,question,response_type\na,Q?,invalid\n",
    "id,question,response_type\na,Q?,yes_no\na,Q?,yes_no\n",
    "id,question,response_type,metadata\na,Q?,yes_no,[]\n",
])
def test_invalid_csv(tmp_path, text):
    path = tmp_path / "bad.csv"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(QuestionnaireError):
        create_drafts(path)


def test_file_errors(tmp_path):
    with pytest.raises(QuestionnaireError):
        create_drafts(tmp_path / "missing.json")
    bad = tmp_path / "bad.csv"
    bad.write_bytes(b"\xff")
    with pytest.raises(QuestionnaireError):
        create_drafts(bad)
    with pytest.raises(QuestionnaireError):
        create_drafts(tmp_path / "unsupported.xlsx")


def test_cli_validates_before_writing_and_refuses_overwrite(tmp_path):
    source, output = tmp_path / "input.json", tmp_path / "drafts.json"
    source.write_text(json.dumps([row(), row(question="")]), encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        main([str(source), str(output)])
    assert exc.value.code == 2 and not output.exists()
    source.write_text(json.dumps([row()]), encoding="utf-8")
    assert main([str(source), str(output)]) == 0
    original = output.read_bytes()
    assert json.loads(original)[0]["answer"] == ""
    with pytest.raises(SystemExit):
        main([str(source), str(output)])
    assert output.read_bytes() == original
    before = source.read_bytes()
    with pytest.raises(SystemExit):
        main([str(source), str(source)])
    assert source.read_bytes() == before


def test_existing_questionnaire_reads_only_input(monkeypatch):
    source = Path("data/eval/clouddesk_questionnaire.json")
    original_open = Path.open
    reads = []

    def input_only(path, *args, **kwargs):
        assert path == source, f"Unexpected file access: {path}"
        reads.append(path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", input_only)
    drafts = create_drafts(source)
    assert reads == [source]
    assert len(drafts) == 100
    assert [d.id for d in drafts] == [f"cdq{i:03}" for i in range(1, 101)]
    assert all(d.answer == "" and d.citations == [] for d in drafts)
    assert all("expected_answer" not in asdict(d) for d in drafts)


def test_module_cli_has_no_model_or_generation_imports(tmp_path):
    source, output = tmp_path / "q.json", tmp_path / "out.json"
    source.write_text(json.dumps([row()]), encoding="utf-8")
    code = '''
import builtins, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.startswith(("google", "chromadb", "sentence_transformers", "trustquery.generation", "trustquery.retrieval", "trustquery.gemini_client", "trustquery.evaluation")):
        raise AssertionError("Unexpected runtime dependency: " + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from trustquery.batch import main
raise SystemExit(main(sys.argv[1:]))
'''
    result = subprocess.run([sys.executable, "-c", code, str(source), str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text(encoding="utf-8"))[0]["status"] == "pending"
