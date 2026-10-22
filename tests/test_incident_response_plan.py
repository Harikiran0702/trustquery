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
PDF = SAMPLE / 'incident_response_plan.pdf'


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
        'incident_plan_builder', ROOT / 'scripts/build_incident_response_plan.py')
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


def test_related_policy_alignment_and_operational_limits():
    source = json.loads(PDF.with_suffix('.json').read_text(encoding='utf-8'))
    text = ' '.join(s['heading'] + ' ' + s['text'] for p in source['pages'] for s in p['sections'])
    for phrase in [
        'fictional 75-person B2B SaaS', 'not operating facts', 'no fixed response guarantee',
        'No dedicated or 24/7 SOC', 'not live services', 'event', 'SEV1 Critical',
        'SEV2 High', 'SEV3 Moderate', 'SEV4 Low', 'chain', 'SHA-256',
        'within 24 hours', 'immediately open an incident', 'where safe',
        'next-business-day review', 'Engineering Lead and CTO approval',
        'backup-specific key grants', 'at least quarterly', 'at least annually',
        'onboarding and annually', 'no universal notification deadline',
        'pre-authorizes no payment', 'minimum 12-month retention',
    ]:
        assert phrase.lower() in text.lower()
    assert set(source['related_documents']) == {
        'information_security_policy.pdf', 'encryption_key_management_policy.pdf',
        'access_control_policy.pdf'}
    related = {name: ' '.join(p['text'] for p in load_pdf(str(SAMPLE / name)))
               for name in source['related_documents']}
    for phrase in ['unique user accounts', 'administrative access', 'every quarter', 'within 24 hours']:
        assert phrase in related['access_control_policy.pdf']
    encryption = related['encryption_key_management_policy.pdf']
    for phrase in ['immediately open an incident', 'where safe', 'at least quarterly',
                   'Engineering Lead and CTO approval', 'next-business-day review']:
        assert phrase in encryption
    governance = related['information_security_policy.pdf']
    for phrase in ['Engineering Lead', 'CTO', 'at least annually', 'onboarding and annually']:
        assert phrase in governance


@pytest.mark.parametrize('query,source,page,answer', [
    ('incident SEV1 SEV2 SEV3 SEV4 severity', PDF.name, 2, 'SEV1'),
    ('evidence custodian SHA-256 acquisition transfer', PDF.name, 3, 'SHA-256'),
    ('SaaS OAuth persistence compromised sessions', PDF.name, 4, 'OAuth'),
    ('cyber extortion sanctions payment attackers', PDF.name, 5, 'payment'),
    ('notification jurisdiction awareness deadline calculation', PDF.name, 5, 'deadline'),
    ('closure lessons root cause corrective actions', PDF.name, 6, 'lessons'),
    ('TLS 1.2 TLS 1.3 plaintext transport', 'encryption_key_management_policy.pdf', 2, 'TLS'),
    ('unique user accounts administrative access', 'access_control_policy.pdf', 1, 'unique user accounts'),
    ('classification Public Internal Confidential Restricted', 'information_security_policy.pdf', 2, 'Restricted'),
])
def test_bm25_across_four_policies(query, source, page, answer):
    documents = [document for name in [PDF.name, 'information_security_policy.pdf',
                 'encryption_key_management_policy.pdf', 'access_control_policy.pdf']
                 for document in load_pdf(str(SAMPLE / name))]
    chunks = chunk_documents(documents)
    results = search_bm25(query, chunks, top_k=3)
    assert any(r['metadata']['source'] == source and r['metadata']['page'] == page
               and answer in r['text'] for r in results)
    for chunk in chunks:
        original = next(d for d in documents if d['source'] == chunk['metadata']['source']
                        and d['page'] == chunk['metadata']['page'])
        assert chunk['text'] in original['text']
        assert isinstance(chunk['metadata']['chunk_index'], int)
