"""Offline document checks; these do not constitute human review or compliance."""
import importlib.util
import json
from pathlib import Path

import pytest
from pypdf import PdfReader

from trustquery.ingestion.chunker import chunk_documents
from trustquery.ingestion.loader import load_pdf
from trustquery.retrieval.bm25_retriever import search_bm25

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data/sample"
PDF = SAMPLE / "information_security_policy.pdf"


def test_source_and_pdf_preserve_sections_provenance_and_page_metadata():
    source = json.loads(PDF.with_suffix(".json").read_text(encoding="utf-8"))
    assert source["synthetic"] is True
    assert source["review_status"] == "requires_user_review"
    assert source["provenance"] == "model_authored"
    assert source["approval_status"] == "not_recorded"
    pages = load_pdf(str(PDF))
    assert len(pages) == len(source["pages"]) == 5
    for number, (page, expected) in enumerate(zip(pages, source["pages"]), 1):
        assert page["page"] == number
        assert page["source"] == PDF.name
        assert expected["heading"] in page["text"]
        assert "SYNTHETIC / DEMO" in page["text"]
        assert f"Page {number}" in page["text"]
        for section in expected["sections"]:
            assert section["heading"] in page["text"]
            assert " ".join(section["text"].split()) in page["text"]
    metadata = PdfReader(PDF).metadata
    assert "SYNTHETIC DEMO" in metadata.title
    assert "fictional" in metadata.author
    assert "requires_user_review" in metadata.subject


def test_build_reproduces_generated_artifact(tmp_path):
    pytest.importorskip("reportlab", reason="Optional PDF authoring tool")
    spec = importlib.util.spec_from_file_location(
        "information_policy_builder", ROOT / "scripts/build_information_security_policy.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    first = builder.build(output_path=tmp_path / "first.pdf").read_bytes()
    second = builder.build(output_path=tmp_path / "second.pdf").read_bytes()
    assert first == second == PDF.read_bytes()


def test_governance_preserves_related_policy_requirements():
    source = json.loads(PDF.with_suffix(".json").read_text(encoding="utf-8"))
    text = " ".join(s["text"] for p in source["pages"] for s in p["sections"])
    for requirement in [
        "unique user accounts", "multi-factor authentication for administrative access",
        "every quarter", "within 24 hours", "TLS 1.2 or TLS 1.3", "AES-256",
        "every 12 months", "at least quarterly", "within 90 days",
        "reviewed monthly", "next business day", "minimum 12-month retention",
    ]:
        assert requirement in text
    assert set(source["related_documents"]) == {
        "access_control_policy.pdf", "encryption_key_management_policy.pdf"}
    for name in source["related_documents"]:
        assert (SAMPLE / name).is_file()
    access = " ".join(" ".join(p["text"].split()) for p in load_pdf(
        str(SAMPLE / "access_control_policy.pdf")))
    for phrase in ["unique user accounts", "administrative access", "every quarter", "within 24 hours"]:
        assert phrase in access


@pytest.mark.parametrize("query,page,answer", [
    ("risk register likelihood impact residual treatment", 2, "risk register"),
    ("classification Public Internal Confidential Restricted", 2, "Restricted"),
    ("vendor criticality agreements exit dependencies", 3, "Vendor management"),
    ("security awareness phishing training onboarding", 3, "training"),
    ("continuity disaster recovery RTO RPO objectives", 4, "RTO"),
    ("incident triage containment severity", 4, "triage"),
    ("exceptions compensating expiry renewal", 5, "90 days"),
    ("internal review findings closure evidence", 5, "closure evidence"),
])
def test_bm25_retrieval_across_three_policies(query, page, answer):
    documents = [document for name in [
        "information_security_policy.pdf", "encryption_key_management_policy.pdf",
        "access_control_policy.pdf"] for document in load_pdf(str(SAMPLE / name))]
    chunks = chunk_documents(documents)
    results = search_bm25(query, chunks, top_k=3)
    assert any(r["metadata"]["source"] == PDF.name
               and r["metadata"]["page"] == page and answer in r["text"] for r in results)
    for chunk in chunks:
        original = next(d for d in documents
                        if d["source"] == chunk["metadata"]["source"]
                        and d["page"] == chunk["metadata"]["page"])
        assert chunk["text"] in original["text"]
        assert isinstance(chunk["metadata"]["chunk_index"], int)
