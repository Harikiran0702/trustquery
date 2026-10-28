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



@pytest.mark.parametrize('page,section,fragments', [
    (0, 0, ('Proposed effective date', 'Planned next review', 'approval not recorded')),
    (0, 1, ('fictional 75-person B2B SaaS', 'has not been approved', 'no SOC 2 report', 'ISO certification', 'legal compliance conclusion')),
    (0, 2, ('confidentiality, integrity and availability', 'measurable security objectives annually', 'CTO approval and progress')),
    (0, 3, ('CTO must set security direction, provide resources and accept risks', 'risk register and internal reviews', 'Managers must coordinate training', 'report suspected exposure promptly')),
    (1, 0, ('at least annually', 'before material service changes', 'before onboarding vendors', 'likelihood, impact', 'residual risk', 'risk ID, owner, treatment, due date, acceptance decision and review date', 'Review open risks quarterly', 'overdue high-impact actions')),
    (1, 1, ('systems, devices, services, data stores and vendors', 'owner, business purpose, classification and dependencies', 'acquired, materially changed or retired', 'no actual asset list')),
    (1, 2, ('Public', 'Internal', 'Confidential', 'Restricted', 'highest applicable class', 'authorized recipients through approved channels', 'Check recipients and limit copies', 'Never put secrets in source code, logs or support exports')),
    (1, 3, ('customer commitments and applicable obligations', 'Record approved periods and any holds', 'verified deletion or secure media disposal', 'no retention periods or backup schedules', 'Public marketing content may be stored unencrypted', 'administrative and authentication traffic')),
    (2, 0, ('unique user accounts', 'multi-factor authentication for administrative access', 'every quarter', 'within 24 hours', 'requests, approvals and revocation records')),
    (2, 1, ('TLS 1.2 or TLS 1.3', 'AES-256', 'managed KMS', 'every 12 months', 'dependent data or backups', 'grant, certificate, secrets and destruction', 'not assumed capabilities')),
    (2, 2, ('at onboarding and annually thereafter', 'phishing, safe sharing, credential protection, incident reporting and classification', 'role-specific guidance', 'attendance, completion and overdue follow-up', 'not evidence of completed training')),
    (2, 3, ('assess each vendor before it handles', 'data access, service criticality', 'CTO must accept material residual risk before use', 'confidentiality, incident communication, access termination and data return or deletion', 'Reassess critical vendors annually', 'material changes or incidents', 'exit dependencies')),
    (3, 0, ('promptly report', 'no live address or staffed response service', 'triage, severity, containment, evidence preservation', 'authorize external communications', 'case by case', 'no universal notification deadline')),
    (3, 1, ('discovery time, affected assets and information, decisions, responders, containment, recovery and corrective actions', 'Restrict evidence access', 'logs and decision history', 'remediation owners and due dates, with CTO review', 'emergency access and key-recovery safeguards')),
    (3, 2, ('critical services, dependencies, disruption scenarios and recovery priorities', 'CTO must approve', 'after impact assessment', 'no numerical targets or availability guarantees', 'responsibilities, dependencies and communication plans', 'at least annually and after material changes', 'gaps and corrective actions')),
    (3, 3, ('backup-specific key grants', 'retention of keys needed for recovery', 'at least quarterly', 'integrity, authorized key access and outcome', 'approved continuity objectives', 'no backup schedules, data retention periods or successful restore history')),
    (4, 0, ('annual internal review', 'report findings and corrective actions to the CTO', 'owners, due dates and closure evidence', 'Review open findings quarterly', 'quarterly access reviews', 'monthly KMS administrative and decrypt audit-event reviews', 'not an independent audit or certification')),
    (4, 1, ('Before deviating', 'affected assets and data', 'rationale, risk, compensating safeguards, owner and expiry', 'CTO must approve or reject', 'within 90 days', 'reviewed monthly', 'fresh assessment and approval', 'next business day', 'authoritative for encryption exceptions and destructive key approvals', 'no active exception')),
    (4, 2, ('access-restricted repository', 'ownership, version history and approved retention schedules', 'minimum 12-month retention for KMS audit logs', 'does not apply to all records', 'no completed evidence or approval signatures')),
    (4, 3, ('follow approved policy requirements', 'report suspected violations', 'proportionate corrective action', 'agreements and employment processes', 'No automatic sanction', 'annually and after material incidents or business or technology changes', 'revisions, reviewer feedback and CTO approval', 'requires_user_review', 'not_recorded')),
])
def test_plain_language_revision_preserves_obligations_and_limits(page, section, fragments):
    source = json.loads(PDF.with_suffix('.json').read_text(encoding='utf-8'))
    assert all(fragment in source['pages'][page]['sections'][section]['text'] for fragment in fragments)


def test_dates_and_metadata_remain_proposals_pending_approval():
    source = json.loads(PDF.with_suffix('.json').read_text(encoding='utf-8'))
    assert 'effective_date' not in source and 'next_review_date' not in source
    assert source['proposed_effective_date'] == '2026-10-22'
    assert source['planned_next_review_date'] == '2027-10-22'
    assert source['date_status'] == 'proposed_dates_approval_not_recorded'
    assert 'approval_status: not_recorded' in PdfReader(PDF).metadata.subject


def test_final_wording_preserves_document_control_vendor_and_record_scope():
    source = json.loads(PDF.with_suffix('.json').read_text(encoding='utf-8'))
    assert len(source['pages']) == 5
    assert sum(len(p['sections']) for p in source['pages']) == 20
    control = source['pages'][0]['sections'][0]['text']
    assert control.count('approval not recorded') == 1
    assert 'Proposed effective date: 2026-10-22' in control
    assert 'Planned next review: 2027-10-22' in control
    vendor = source['pages'][2]['sections'][3]['text']
    for phrase in ('business owner must assess each vendor before it handles',
                   'data access, service criticality, security requirements',
                   'available assurance evidence and identified risks',
                   'Engineering Lead must review the assessment',
                   'CTO must accept material residual risk before use',
                   'Agreements must set appropriate confidentiality',
                   'incident communication, access termination and data return or deletion',
                   'Reassess critical vendors annually and after material changes or incidents',
                   'Track findings and exit dependencies',
                   'Do not presume vendor certifications or assurances'):
        assert phrase in vendor
    records = source['pages'][4]['sections'][2]['text']
    for phrase in ('risk register', 'asset inventory', 'assessments and approvals',
                   'training records', 'vendor reviews', 'access reviews',
                   'incidents', 'recovery exercises', 'findings', 'exception tickets',
                   'access-restricted repository', 'ownership, version history',
                   'approved retention schedules',
                   'minimum 12-month retention for KMS audit logs',
                   'that period does not apply to all records',
                   'no completed evidence or approval signatures'):
        assert phrase in records
