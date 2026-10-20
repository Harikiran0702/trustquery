"""Single-chunk NLI verification, independent of the evaluation pipeline.

The default English MiniLM model was trained on SNLI/MultiNLI, not security
policies. Scores are model probabilities, not calibrated factual confidence.
No model is loaded at import time. See the model card for the label order:
https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768
"""

from dataclasses import dataclass
import math
from typing import Protocol, Sequence


DEFAULT_NLI_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"
NLI_LABELS = ("contradiction", "entailment", "neutral")


class NLIBackend(Protocol):
    """Return three raw logits in NLI_LABELS order for one ordered pair."""

    def logits(self, premise: str, hypothesis: str) -> Sequence[float]: ...


@dataclass(frozen=True)
class NLIResult:
    label: str
    contradiction: float
    entailment: float
    neutral: float


class _MiniLMBackend:
    def __init__(self, model):
        self.model = model

    def logits(self, premise: str, hypothesis: str) -> Sequence[float]:
        # Include special tokens; reject oversize input instead of silently
        # dropping evidence (or part of the claim) during CrossEncoder.predict.
        tokens = self.model.tokenizer(
            premise, hypothesis, truncation=False, add_special_tokens=True,
        )
        if len(tokens["input_ids"]) > self.model.max_length:
            raise ValueError("NLI pair exceeds model token limit; use a smaller chunk")
        return self.model.predict(
            [(premise, hypothesis)], apply_softmax=False,
            show_progress_bar=False,
        )[0]


def load_nli_backend(*, local_files_only: bool = True) -> NLIBackend:
    """Load the fixed three-class model on CPU for local inference.

    Cache model files first, or explicitly pass local_files_only=False to allow
    a download. Reuse the returned backend across calls. No text is sent to an
    inference service. Requires the existing sentence-transformers dependency.
    """
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(
        DEFAULT_NLI_MODEL, device="cpu", max_length=512,
        local_files_only=local_files_only,
    )
    return _MiniLMBackend(model)


def verify_claim(claim: str, evidence: str, backend: NLIBackend) -> NLIResult:
    """Compare evidence (premise) to claim (hypothesis), without aggregation.

    Inject a backend for offline tests. Custom backends must return raw logits
    in contradiction/entailment/neutral order, regardless of their native order.
    Ties resolve in NLI_LABELS order; no support threshold is applied here.
    """
    if not claim.strip() or not evidence.strip():
        raise ValueError("claim and evidence must both be non-empty")
    logits = tuple(float(value) for value in backend.logits(evidence, claim))
    if len(logits) != 3 or not all(math.isfinite(value) for value in logits):
        raise ValueError("NLI backend must return exactly three finite logits")
    weights = [math.exp(value - max(logits)) for value in logits]
    total = sum(weights)
    probabilities = [value / total for value in weights]
    label = NLI_LABELS[max(range(3), key=logits.__getitem__)]
    return NLIResult(label, *probabilities)
