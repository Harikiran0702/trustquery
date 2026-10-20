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
    assert result.score == 1
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
