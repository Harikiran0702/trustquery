from unittest.mock import Mock, call

import pytest

from trustquery.evaluation.faithfulness import answer_faithfulness


def chunk(text="Evidence.", **metadata):
    return {"text": text, "metadata": metadata}


def test_supported_and_unsupported_claims():
    backend = Mock()
    backend.logits.side_effect = [[0, 5, 0], [5, 0, 0], [0, 0, 5]]
    result = answer_faithfulness("First. Second. Third.", [chunk()], backend)
    assert result.score == pytest.approx(1 / 3)
    assert [c.claim for c in result.claims] == ["First.", "Second.", "Third."]
    assert [c.supported for c in result.claims] == [True, False, False]
    assert [c.evidence[0].result.label for c in result.claims] == [
        "entailment", "contradiction", "neutral",
    ]


def test_multiple_chunks_remain_separate_with_conflicts_and_provenance():
    backend = Mock()
    backend.logits.side_effect = [[5, 0, 0], [0, 5, 0], [0, 0, 5]]
    chunks = [chunk("Opposite.", source="a.pdf", page=1, chunk_index=0),
              chunk("Support.", source="b.pdf", page=2, chunk_index=3),
              chunk("Unrelated.")]
    result = answer_faithfulness("Claim.", chunks, backend)
    assert result.score == 0
    assert result.claims[0].conflicted
    assert not result.claims[0].supported
    assert backend.logits.call_args_list == [
        call("Opposite.", "Claim."), call("Support.", "Claim."),
        call("Unrelated.", "Claim."),
    ]
    evidence = result.claims[0].evidence
    assert [e.evidence_index for e in evidence] == [0, 1, 2]
    assert [e.metadata for e in evidence] == [c["metadata"] for c in chunks]
    assert [e.result.label for e in evidence] == [
        "contradiction", "entailment", "neutral",
    ]
    chunks[1]["metadata"]["page"] = 99
    assert evidence[1].metadata["page"] == 2


@pytest.mark.parametrize("answer", ["", " \n ", "Yes!", "Source: policy.pdf"])
def test_no_claims(answer):
    backend = Mock()
    result = answer_faithfulness(answer, [chunk()], backend)
    assert result.score == 0
    assert result.claims == ()
    backend.logits.assert_not_called()


def test_no_evidence():
    backend = Mock()
    result = answer_faithfulness("Claim.", [], backend)
    assert result.score == 0
    assert not result.claims[0].supported
    assert not result.claims[0].conflicted
    assert result.claims[0].evidence == ()
    backend.logits.assert_not_called()


@pytest.mark.parametrize("bad", [None, "text", {}, {"text": 1},
    {"text": " "}, {"text": "ok", "metadata": None},
    {"text": "ok", "metadata": []}])
@pytest.mark.parametrize("answer", ["Claim.", ""])
def test_malformed_chunks_fail_before_inference(bad, answer):
    backend = Mock()
    with pytest.raises(ValueError, match="evidence chunk 1"):
        answer_faithfulness(answer, [chunk(), bad], backend)
    backend.logits.assert_not_called()


@pytest.mark.parametrize("bad", [None, "text", {}, 5])
def test_invalid_evidence_container(bad):
    with pytest.raises(TypeError, match="sequence"):
        answer_faithfulness("Claim.", bad, Mock())


def test_invalid_answer():
    with pytest.raises(TypeError, match="answer"):
        answer_faithfulness(None, [], Mock())


@pytest.mark.parametrize("error", [ValueError("NLI pair exceeds model token limit"),
    RuntimeError("backend unavailable")])
def test_errors_abort_even_after_entailment(error):
    backend = Mock()
    backend.logits.side_effect = [[0, 5, 0], error]
    with pytest.raises(RuntimeError, match="claim 0, evidence chunk 1") as caught:
        answer_faithfulness("Claim.", [chunk(), chunk()], backend)
    assert caught.value.__cause__ is error


def test_invalid_logits_not_counted_as_unsupported():
    backend = Mock()
    backend.logits.return_value = [float("nan"), 0, 0]
    with pytest.raises(RuntimeError, match="three finite logits"):
        answer_faithfulness("Claim.", [chunk()], backend)


def test_metadata_optional_and_tied_logits_do_not_imply_support():
    backend = Mock()
    backend.logits.return_value = [0, 0, 0]
    result = answer_faithfulness("Claim.", [{"text": "Evidence."}], backend)
    assert result.score == 0
    assert result.claims[0].evidence[0].metadata == {}


@pytest.mark.parametrize("labels, supported, conflicted", [
    ([1, 0], False, True),
    ([0, 1], False, True),
    ([1, 2, 1, 0], False, True),
    ([0, 2, 0, 1], False, True),
    ([1, 1], True, False),
    ([1, 2], True, False),
    ([2, 1], True, False),
    ([0, 0], False, False),
    ([0, 2], False, False),
    ([2, 2], False, False),
])
def test_chunk_label_combinations(labels, supported, conflicted):
    backend = Mock()
    backend.logits.side_effect = [
        [5 if i == label else 0 for i in range(3)] for label in labels
    ]
    result = answer_faithfulness(
        "Claim.", [chunk(f"Evidence {i}.") for i in range(len(labels))], backend,
    )
    claim = result.claims[0]
    assert claim.supported is supported
    assert claim.conflicted is conflicted
    assert result.score == float(supported)
    assert backend.logits.call_count == len(labels)
    assert [e.result.label for e in claim.evidence] == [
        ("contradiction", "entailment", "neutral")[label] for label in labels
    ]


def test_conflicts_are_per_claim_and_excluded_from_aggregate():
    backend = Mock()
    labels = [1, 0, 2, 1, 2, 1, 0, 2, 0, 2, 2, 2]
    backend.logits.side_effect = [
        [5 if i == label else 0 for i in range(3)] for label in labels
    ]
    chunks = [
        chunk(f"Evidence {i}.", source=f"policy-{i}", page=i + 1,
              chunk_id=f"access-{i}", fictional=True, custom="preserved")
        for i in range(3)
    ]
    result = answer_faithfulness(
        "Conflicted. Supported. Contradicted. Neutral.", chunks, backend,
    )
    assert result.score == 0.25
    assert [c.supported for c in result.claims] == [False, True, False, False]
    assert [c.conflicted for c in result.claims] == [True, False, False, False]
    for claim_index, claim in enumerate(result.claims):
        assert [e.evidence_index for e in claim.evidence] == [0, 1, 2]
        assert [e.metadata for e in claim.evidence] == [c["metadata"] for c in chunks]
        assert [e.result.label for e in claim.evidence] == [
            ("contradiction", "entailment", "neutral")[label]
            for label in labels[claim_index * 3:claim_index * 3 + 3]
        ]
        for evidence in claim.evidence:
            probabilities = (evidence.result.contradiction,
                             evidence.result.entailment, evidence.result.neutral)
            assert sum(probabilities) == pytest.approx(1)
            assert max(probabilities) == pytest.approx(0.986703291042268)
    chunks[0]["metadata"]["source"] = "changed"
    assert all(c.evidence[0].metadata["source"] == "policy-0" for c in result.claims)
