import json
from unittest.mock import Mock, call, patch

import pytest

from trustquery.evaluation.evaluator import evaluate_generation


def item(index=0, should_answer=True):
    return dict(id=str(index), question=str(index), should_answer=should_answer,
                expected_source="policy.pdf", expected_page=1)


def chunk(text="Evidence.", page=1):
    return dict(text=text, metadata=dict(source="policy.pdf", page=page,
                                       chunk_index=page, custom="kept"))


@pytest.mark.parametrize("labels,score,supported,conflicted", [
    ([1], 1.0, True, False),
    ([0], 0.0, False, False),
    ([2], 0.0, False, False),
    ([1, 0], 0.0, False, True),
    ([0, 1], 0.0, False, True),
])
def test_verdicts_and_provenance(labels, score, supported, conflicted):
    backend = Mock()
    backend.logits.side_effect = [
        [5 if i == label else 0 for i in range(3)] for label in labels
    ]
    chunks = [chunk(f"Evidence {i}.", i + 1) for i in range(len(labels))]
    retrieve, generate = Mock(return_value=chunks), Mock(return_value="Claim.")
    report = evaluate_generation([item()], retrieve, generate, nli_backend=backend)
    row = report["results"][0]
    assert row["faithfulness_score"] == report["faithfulness_score"] == score
    assert row["faithfulness_status"] == "scored"
    claim = row["faithfulness_claims"][0]
    assert claim["claim"] == "Claim."
    assert claim["supported"] is supported
    assert claim["conflicted"] is conflicted
    for i, evidence in enumerate(claim["evidence"]):
        assert evidence["evidence_index"] == i
        assert evidence["metadata"] == chunks[i]["metadata"]
        verdict = evidence["result"]
        assert verdict["label"] == ("contradiction", "entailment", "neutral")[labels[i]]
        assert sum(verdict[k] for k in ("contradiction", "entailment", "neutral")) == pytest.approx(1)
    assert backend.logits.call_args_list == [call(c["text"], "Claim.") for c in chunks]
    retrieve.assert_called_once_with("0")
    generate.assert_called_once_with(question="0", retrieved_chunks=chunks)
    chunks[0]["metadata"]["page"] = 99
    assert claim["evidence"][0]["metadata"]["page"] == 1
    json.dumps(report)


@pytest.mark.parametrize("answer,chunks,status", [
    (" ", [chunk()], "empty_answer"),
    ("Insufficient evidence in the provided documents.", [chunk()], "abstained"),
    ("Not enough evidence.", [], "abstained"),
    ("Cannot answer.", [chunk()], "abstained"),
    ("Yes!", [chunk()], "no_claims"),
    ("Source: policy.pdf, Page: 1", [chunk()], "no_claims"),
    ("Claim.", [], "missing_evidence"),
])
def test_unverified_cases_do_not_raise_aggregate(answer, chunks, status):
    backend = Mock()
    backend.logits.return_value = [0, 5, 0]
    report = evaluate_generation(
        [item(0), item(1)], lambda q: [chunk()] if q == "0" else chunks,
        lambda **kw: "Claim." if kw["question"] == "0" else answer,
        nli_backend=backend,
    )
    row = report["results"][1]
    assert row["faithfulness_status"] == status
    assert row["faithfulness_score"] == 0
    assert report["faithfulness_score"] == 0.5
    assert report["faithfulness_answerable_count"] == 2
    assert report["faithfulness_scored_count"] == 1
    assert report["faithfulness_unscored_count"] == 1
    backend.logits.assert_called_once()
    if status == "missing_evidence":
        assert row["faithfulness_claims"] == [dict(
            claim="Claim.", supported=False, conflicted=False, evidence=(),
        )]
    else:
        assert row["faithfulness_claims"] == []


def test_backward_compatibility_and_no_implicit_loader():
    dataset = [item(), item(1, False)]
    retrieve = lambda q: [chunk()]
    generate = lambda **kw: ("Claim.\nSource: policy.pdf, Page: 1"
                             if kw["question"] == "0" else "Cannot answer.")
    with patch("trustquery.evaluation.nli.load_nli_backend", side_effect=AssertionError):
        old = evaluate_generation(dataset, retrieve, generate)
        backend = Mock()
        backend.logits.return_value = [0, 5, 0]
        new = evaluate_generation(dataset, retrieve, generate, nli_backend=backend)
    for key in ("abstention_accuracy", "citation_coverage", "citation_correctness"):
        assert old[key] == new[key] == 1
    for before, after in zip(old["results"], new["results"]):
        assert {k: v for k, v in before.items() if not k.startswith("faithfulness_")} == {
            k: v for k, v in after.items() if not k.startswith("faithfulness_")}
        assert before["faithfulness_status"] == "disabled"
        assert before["faithfulness_score"] is None
    assert old["faithfulness_score"] is None
    assert old["faithfulness_unscored_count"] == 1
    assert new["faithfulness_score"] == 1
    assert new["faithfulness_answerable_count"] == 1


@pytest.mark.parametrize("dataset", [[], [item(0, False)]])
def test_no_answerable_cases(dataset):
    backend = Mock()
    backend.logits.return_value = [0, 5, 0]
    report = evaluate_generation(dataset, lambda q: [chunk()],
                                 lambda **kw: "Claim.", nli_backend=backend)
    assert report["faithfulness_score"] is None
    assert report["faithfulness_answerable_count"] == 0
    assert report["faithfulness_scored_count"] == 0
    assert report["faithfulness_unscored_count"] == 0
    if dataset:
        assert report["results"][0]["faithfulness_score"] == 1


@pytest.mark.parametrize("chunks,error", [
    ([chunk()], RuntimeError("offline failure")),
    ([dict(text=" ")], None),
])
def test_failures_propagate(chunks, error):
    backend = Mock()
    backend.logits.side_effect = error
    with pytest.raises((RuntimeError, ValueError)):
        evaluate_generation([item()], lambda q: chunks, lambda **kw: "Claim.",
                            nli_backend=backend)
