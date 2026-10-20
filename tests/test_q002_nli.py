"""q002 diagnostic: human expectations and historical model errors stay separate.

Offline tests inject synthetic logits; they do not validate MiniLM semantics.
Run the real cached model explicitly (no downloads):
    TRUSTQUERY_RUN_NLI_DIAGNOSTIC=1 python -m pytest tests/test_q002_nli.py -rxX
In PowerShell set $env:TRUSTQUERY_RUN_NLI_DIAGNOSTIC = "1" first.
A missing/broken model cache is an error, not an expected semantic failure.
"""

import json
import os
from pathlib import Path
from unittest.mock import Mock, call

import pytest

from trustquery.evaluation.faithfulness import answer_faithfulness, extract_claims
from trustquery.evaluation.nli import NLI_LABELS, load_nli_backend, verify_claim
from trustquery.ingestion.loader import load_pdf


@pytest.fixture
def q002():
    return json.loads(
        Path(__file__).with_name("q002_nli_fixture.json").read_text(encoding="utf-8-sig")
    )


def test_q002_snapshot_matches_policy_and_claim(q002):
    pages = load_pdf("data/sample/access_control_policy.pdf")
    assert extract_claims(q002["cached_answer"]) == [q002["claim"]]
    assert [p["human_expected_label"] for p in q002["pages"]] == ["entailment", "neutral"]
    assert [p["previously_observed_minilm_label"] for p in q002["pages"]] == [
        "entailment", "contradiction",
    ]
    for snapshot, page in zip(q002["pages"], pages, strict=True):
        assert snapshot["text"] == page["text"]
        assert snapshot["metadata"]["source"] == page["source"]
        assert snapshot["metadata"]["page"] == page["page"]


@pytest.mark.parametrize("label_field,supported,conflicted", [
    ("human_expected_label", True, False),
    ("previously_observed_minilm_label", False, True),
])
def test_q002_offline_verdict_replay(q002, label_field, supported, conflicted):
    # Synthetic logits replay labels only, not measured model probabilities.
    backend = Mock()
    backend.logits.side_effect = [
        [5 if label == page[label_field] else 0 for label in NLI_LABELS]
        for page in q002["pages"]
    ]
    result = answer_faithfulness(q002["cached_answer"], q002["pages"], backend)
    assert len(result.claims) == 1
    claim = result.claims[0]
    assert claim.claim == q002["claim"]
    assert claim.supported is supported
    assert claim.conflicted is conflicted
    assert result.score == float(supported)
    assert [e.result.label for e in claim.evidence] == [
        p[label_field] for p in q002["pages"]
    ]
    assert [e.metadata for e in claim.evidence] == [p["metadata"] for p in q002["pages"]]
    assert backend.logits.call_args_list == [
        call(p["text"], q002["claim"]) for p in q002["pages"]
    ]


@pytest.mark.skipif(
    os.environ.get("TRUSTQUERY_RUN_NLI_DIAGNOSTIC") != "1",
    reason="opt-in real-model diagnostic; normal unit tests are offline",
)
@pytest.mark.parametrize("page_index", [0, 1], ids=["page1-entailment", "page2-neutral"])
def test_q002_cached_minilm_against_human_expectation(q002, page_index):
    backend = load_nli_backend(local_files_only=True)
    page = q002["pages"][page_index]
    result = verify_claim(q002["claim"], page["text"], backend)
    if page_index == 1 and result.label == "contradiction":
        pytest.xfail(
            "Known MiniLM false contradiction: human expectation is neutral; "
            "reviews and termination revocation do not negate unique accounts"
        )
    assert result.label == page["human_expected_label"], result
