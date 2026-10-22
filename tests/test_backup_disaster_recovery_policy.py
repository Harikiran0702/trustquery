"""Offline sample checks; passing tests are not human approval or compliance evidence."""
import importlib.util
import json
from pathlib import Path

import pytest
from pypdf import PdfReader

from trustquery.ingestion.chunker import chunk_documents
from trustquery.ingestion.loader import load_pdf
from trustquery.retrieval.bm25_retriever import search_bm25

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / 'data/sample'
PDF = SAMPLE / 'backup_disaster_recovery_policy.pdf'


def test_source_typography_provenance_and_searchable_pages():
    raw = PDF.with_suffix('.json').read_bytes()
    source = json.loads(raw)
    assert source['synthetic'] is True
    assert source['review_status'] == 'requires_user_review'
    assert source['provenance'] == 'model_authored'
    assert source['approval_status'] == 'not_recorded'
    assert source['owner'] == 'Engineering Lead'
    assert source['approver'] == 'CTO'
    assert raw.decode('utf-8').replace('\r\n', '\n') == (
        json.dumps(source, indent=2, ensure_ascii=True) + '\n')
    pages = load_pdf(str(PDF))
    assert len(pages) == len(source['pages']) == 6
    for number, (actual, expected) in enumerate(zip(pages, source['pages']), 1):
        assert actual['source'] == PDF.name
        assert actual['page'] == number
        assert expected['heading'] in actual['text']
        assert 'SYNTHETIC / DEMO' in actual['text']
        assert f'Page {number}' in actual['text']
        for section in expected['sections']:
            assert section['heading'] in actual['text']
            assert section['text'].isascii()
            assert section['text'] == section['text'].strip()
            assert '  ' not in section['text']
            assert section['text'] in actual['text']
    metadata = PdfReader(PDF).metadata
    assert 'SYNTHETIC DEMO' in metadata.title
    assert 'fictional' in metadata.author
    assert 'requires_user_review' in metadata.subject


def get_builder():
    pytest.importorskip('reportlab', reason='Optional PDF authoring tool')
    spec = importlib.util.spec_from_file_location(
        'incident_plan_builder', ROOT / 'scripts/build_backup_disaster_recovery_policy.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rebuild_is_byte_identical(tmp_path):
    builder = get_builder()
    first = builder.build(output_path=tmp_path / 'first.pdf').read_bytes()
    second = builder.build(output_path=tmp_path / 'second.pdf').read_bytes()
    assert first == second == PDF.read_bytes()


def test_builder_preserves_literal_markup_and_spacing(tmp_path):
    builder = get_builder()
    source = json.loads(PDF.with_suffix('.json').read_text(encoding='utf-8'))
    literal = 'Evidence <original> & copy; "quoted" path C:\\case\\item.'
    source['pages'][0]['sections'][0]['text'] = literal
    edited = tmp_path / 'edited.json'
    edited.write_text(json.dumps(source), encoding='utf-8')
    output = builder.build(source_path=edited, output_path=tmp_path / 'edited.pdf')
    assert literal in load_pdf(str(output))[0]['text']



def source_text():
    source = json.loads(PDF.with_suffix('.json').read_text(encoding='utf-8'))
    return source, ' '.join(s['text'] for p in source['pages'] for s in p['sections'])


def test_related_policy_alignment_and_operational_limits():
    source, text = source_text()
    assert set(source['related_documents']) == {
        'access_control_policy.pdf', 'encryption_key_management_policy.pdf',
        'information_security_policy.pdf', 'incident_response_plan.pdf'}
    related = {name: ' '.join(p['text'] for p in load_pdf(str(SAMPLE / name)))
               for name in source['related_documents']}
    contracts = {
        'access_control_policy.pdf': ['unique user accounts', 'administrative access',
                                     'every quarter', 'within 24 hours'],
        'encryption_key_management_policy.pdf': ['AES-256', 'TLS 1.2 or TLS 1.3',
            'at least quarterly', 'authorized key access', 'recovery outcome',
            'where safe', 'next-business-day review', 'Engineering Lead and CTO approval',
            'at least every 12 months', 'no default decrypt permission'],
        'information_security_policy.pdf': ['at least annually and after material changes',
            'CTO approves', 'review open risks quarterly'],
        'incident_response_plan.pdf': ['tenant separation', 'Engineering Lead verifies containment',
            'CTO approves material residual risk', 'no universal notification deadline'],
    }
    for name, phrases in contracts.items():
        for phrase in phrases:
            assert phrase in related[name], (name, phrase)
    for phrase in ['fictional 75-person B2B SaaS', 'not operating facts',
        'backup-specific key grants separate from ordinary application access',
        'at least quarterly', 'at least annually and after material changes',
        'TBD pending BIA/business and contractual review', 'no numerical targets',
        'Engineering Lead and CTO approval', 'next-business-day review',
        'within 90 days', 'reviewed monthly', 'minimum 12-month retention for KMS audit logs']:
        assert phrase in text, phrase


@pytest.mark.parametrize('heading, required, forbidden', [
    ('Proposed frequency and retention - TBD', ['TBD', 'No daily, weekly', 'before deployment'],
     ['backups run daily', 'retain backups for 30 days']),
    ('Business impact analysis and objectives', ['CTO approves', 'no numerical targets'],
     ['RTO: 4 hours', 'RPO: 24 hours']),
    ('Proposed isolation and immutability controls', ['No immutability', 'asserted as implemented'],
     ['backups are immutable', 'guaranteed ransomware resistance']),
    ('Negative recovery scenarios and remediation', ['denied decrypt grants', 'stop unsafe restoration',
      'Never bypass authorization', 'cross-tenant access attempts'], ['ignore integrity failures']),
    ('Annual continuity exercises are separate', ['does not replace', 'No exercise or restoration is claimed completed'],
     ['quarterly tabletop replaces restore']),
    ('Key versions and recoverability', ['full lifetime', 'Never destroy', 'holds'],
     ['delete old keys immediately']),
    ('Tenant integrity and service gates', ['stop release', 'rollback', 'tenant boundaries'],
     ['restore directly to production without validation']),
])
def test_negative_scenarios_and_nonclaims(heading, required, forbidden):
    source, _ = source_text()
    text = next(s['text'] for p in source['pages'] for s in p['sections'] if s['heading'] == heading)
    for phrase in required:
        assert phrase in text
    for phrase in forbidden:
        assert phrase not in text


@pytest.mark.parametrize('query,source,page,answer', [
    ('backup inventory classification coverage exclusions', PDF.name, 2, 'inventory'),
    ('BIA RTO RPO loss tolerance objectives', PDF.name, 2, 'BIA'),
    ('backup frequency retention TBD schedules', PDF.name, 2, 'TBD'),
    ('immutability isolation offline deletion ransomware resistance', PDF.name, 3, 'immutability'),
    ('KMS older key versions retained backups rotation', PDF.name, 3, 'versions'),
    ('disaster declaration alternate communications contacts', PDF.name, 4, 'declaration'),
    ('restoration dependency sequence queues indexes', PDF.name, 4, 'sequence'),
    ('tenant integrity transactional consistency replayed events', PDF.name, 4, 'tenant'),
    ('quarterly encrypted restore tests authorized key access', PDF.name, 5, 'quarterly'),
    ('annual continuity tabletop exercises separate', PDF.name, 5, 'tabletop'),
    ('exceptions expiry renewal monthly 90 days', PDF.name, 6, 'Exceptions'),
    ('TLS 1.2 TLS 1.3 plaintext transport', 'encryption_key_management_policy.pdf', 2, 'TLS'),
    ('unique user accounts administrative access', 'access_control_policy.pdf', 1, 'unique user accounts'),
    ('classification Public Internal Confidential Restricted', 'information_security_policy.pdf', 2, 'Restricted'),
    ('incident SEV1 SEV2 SEV3 SEV4 severity', 'incident_response_plan.pdf', 2, 'SEV1'),
])
def test_bm25_across_five_policies(query, source, page, answer):
    policy, _ = source_text()
    documents = [d for name in [PDF.name] + policy['related_documents']
                 for d in load_pdf(str(SAMPLE / name))]
    chunks = chunk_documents(documents)
    results = search_bm25(query, chunks, top_k=3)
    assert any(r['metadata']['source'] == source and r['metadata']['page'] == page
               and answer in r['text'] for r in results)
    for chunk in chunks:
        original = next(d for d in documents if d['source'] == chunk['metadata']['source']
                        and d['page'] == chunk['metadata']['page'])
        assert chunk['text'] in original['text']
        assert isinstance(chunk['metadata']['chunk_index'], int)
