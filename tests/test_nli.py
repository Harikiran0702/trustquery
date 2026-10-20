import math
from unittest.mock import Mock, patch

import pytest

from trustquery.evaluation.nli import (
    DEFAULT_NLI_MODEL,
    NLI_LABELS,
    load_nli_backend,
    verify_claim,
)


@pytest.mark.parametrize("evidence,logits,label", [
    ("MFA is required.", [0, 4, 0], "entailment"),
    ("MFA is not required.", [4, 0, 0], "contradiction"),
    ("Backups run daily.", [0, 0, 4], "neutral"),
])
def test_verify_claim_labels_and_pair_direction(evidence, logits, label):
    backend = Mock()
    backend.logits.return_value = logits
    result = verify_claim("MFA is required.", evidence, backend)
    assert result.label == label
    backend.logits.assert_called_once_with(evidence, "MFA is required.")
    assert sum(getattr(result, name) for name in NLI_LABELS) == pytest.approx(1)
    assert getattr(result, label) == pytest.approx(math.exp(4) / (math.exp(4) + 2))


@pytest.mark.parametrize("index,label", list(enumerate(NLI_LABELS)))
def test_model_label_mapping(index, label):
    # Explicit expected native order; do not infer semantics from generic LABEL_n.
    assert NLI_LABELS == ("contradiction", "entailment", "neutral")
    backend = Mock()
    backend.logits.return_value = [1000 if i == index else -1000 for i in range(3)]
    result = verify_claim("claim", "evidence", backend)
    assert result.label == label
    assert getattr(result, label) == 1


@pytest.mark.parametrize("claim,evidence", [("", "text"), ("text", " \n ")])
def test_empty_input_does_not_call_backend(claim, evidence):
    backend = Mock()
    with pytest.raises(ValueError, match="non-empty"):
        verify_claim(claim, evidence, backend)
    backend.logits.assert_not_called()


@pytest.mark.parametrize("logits", [[], [1, 2], [1, 2, 3, 4], [0, float("nan"), 1], [0, float("inf"), 1]])
def test_invalid_logits(logits):
    backend = Mock()
    backend.logits.return_value = logits
    with pytest.raises(ValueError, match="three finite logits"):
        verify_claim("claim", "evidence", backend)


def test_tie_is_deterministic():
    backend = Mock()
    backend.logits.return_value = [0, 0, 0]
    result = verify_claim("claim", "evidence", backend)
    assert result.label == "contradiction"
    assert result.entailment == pytest.approx(1 / 3)


@pytest.mark.parametrize("local_files_only", [True, False])
def test_loader_and_adapter_without_model_download(local_files_only):
    with patch("sentence_transformers.CrossEncoder") as constructor:
        model = constructor.return_value
        model.max_length = 512
        model.tokenizer.return_value = {"input_ids": list(range(512))}
        model.predict.return_value = [[0, 3, 0]]
        backend = load_nli_backend(local_files_only=local_files_only)
        constructor.assert_called_once_with(
            DEFAULT_NLI_MODEL, device="cpu", max_length=512,
            local_files_only=local_files_only,
        )
        assert verify_claim("claim", "evidence", backend).label == "entailment"
        model.tokenizer.assert_called_once_with(
            "evidence", "claim", truncation=False, add_special_tokens=True,
        )
        model.predict.assert_called_once_with(
            [("evidence", "claim")], apply_softmax=False, show_progress_bar=False,
        )


def test_loader_defaults_to_cached_files_and_rejects_long_pairs():
    with patch("sentence_transformers.CrossEncoder") as constructor:
        model = constructor.return_value
        model.max_length = 512
        model.tokenizer.return_value = {"input_ids": list(range(513))}
        backend = load_nli_backend()
        assert constructor.call_args.kwargs["local_files_only"] is True
        with pytest.raises(ValueError, match="token limit"):
            verify_claim("claim", "evidence", backend)
        model.predict.assert_not_called()


def test_backend_errors_are_not_reported_as_neutral():
    backend = Mock()
    backend.logits.side_effect = RuntimeError("inference failed")
    with pytest.raises(RuntimeError, match="inference failed"):
        verify_claim("claim", "evidence", backend)
