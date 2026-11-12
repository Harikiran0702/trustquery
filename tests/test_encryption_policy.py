"""Offline ingestion/retrieval checks, not semantic approval or compliance tests."""
import json
from pathlib import Path

import pytest
from pypdf import PdfReader

from trustquery.ingestion.loader import load_pdf
from trustquery.ingestion.chunker import chunk_documents
from trustquery.retrieval.bm25_retriever import search_bm25

SAMPLE = Path(__file__).resolve().parents[1] / 'data/sample'
PDF = SAMPLE / 'encryption_key_management_policy.pdf'


def test_policy_extracts_all_source_sections_and_metadata():
    source = json.loads(PDF.with_suffix('.json').read_text(encoding='utf-8'))
    assert source['synthetic'] is True
    assert source['review_status'] == 'requires_user_review'
    assert source['provenance'] == 'model_authored'
    assert source['approval_status'] == 'not_recorded'
    pages = load_pdf(str(PDF))
    assert len(pages) == len(source['pages']) == 4
    for number, (page, expected) in enumerate(zip(pages, source['pages']), 1):
        assert page['page'] == number
        assert page['source'] == PDF.name
        assert expected['heading'] in page['text']
        assert 'SYNTHETIC / DEMO' in page['text']
        assert f'Page {number}' in page['text']
        for section in expected['sections']:
            assert section['heading'] in page['text']
            assert ' '.join(section['text'].split()) in page['text']
    metadata = PdfReader(PDF).metadata
    assert 'SYNTHETIC DEMO' in metadata.title
    assert 'fictional' in metadata.author


@pytest.mark.parametrize('query,page,answer', [
    ('TLS transport certificate hostname expiry', 2, 'TLS 1.2'),
    ('AES-256 storage databases volumes', 2, 'AES-256'),
    ('key rotation versions 12 months', 3, '12 months'),
    ('compromise revocation destruction', 3, 'compromise'),
    ('backup restore quarterly', 4, 'quarterly'),
    ('exceptions expiry 90 days', 4, '90 days'),
])
def test_policy_bm25_retrieval_preserves_page_citations(query, page, answer):
    documents = load_pdf(str(PDF)) + load_pdf(str(SAMPLE / 'access_control_policy.pdf'))
    chunks = chunk_documents(documents)
    results = search_bm25(query, chunks, top_k=3)
    assert any(result['metadata']['source'] == PDF.name
               and result['metadata']['page'] == page
               and answer in result['text'] for result in results)
    for chunk in chunks:
        original = next(d for d in documents if d['source'] == chunk['metadata']['source']
                        and d['page'] == chunk['metadata']['page'])
        assert chunk['text'] in original['text']
        assert isinstance(chunk['metadata']['chunk_index'], int)


def test_reproducible_build(tmp_path):
    import importlib.util
    pytest.importorskip('reportlab', reason='Optional PDF authoring tool')
    spec = importlib.util.spec_from_file_location(
        'encryption_builder', SAMPLE.parents[1] / 'scripts/build_encryption_policy.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    first = builder.build(output_path=tmp_path / 'one.pdf').read_bytes()
    second = builder.build(output_path=tmp_path / 'two.pdf').read_bytes()
    assert first == second == PDF.read_bytes()


@pytest.mark.parametrize('page,section,fragments', [
    (0, 2, ('Public marketing content may be stored unencrypted',)),
    (0, 3, ('Customer-managed keys and end-to-end encryption are not assumed',)),
    (1, 0, ('TLS 1.2 or TLS 1.3', 'carrying scoped information', 'plaintext fallbacks')),
    (1, 1, ('at least 30 days', 'without bypassing verification')),
    (1, 2, ('AES-256', 'scoped log archives', 'full-disk encryption', 'unencrypted removable media')),
    (1, 3, ('Salt and hash passwords', 'do not reversibly encrypt', 'authorized application processes')),
    (2, 0, ('envelope encryption', 'non-exportable', 'Separate production keys')),
    (2, 1, ('no default decrypt permission', 'Engineering Lead approves', 'every quarter', 'within 24 hours')),
    (2, 2, ('symmetric key-encryption keys at least every 12 months', 'where supported',
             'only for data or backups still within retention', 'does not re-encrypt historical data')),
    (2, 3, ('where safe', 'backup recoverability', 'CTO must authorize', 'next-business-day review',
             'Engineering Lead and CTO approval')),
    (3, 0, ('AES-256', 'TLS 1.2 or TLS 1.3', 'full lifetime', 'legal or incident holds',
             'representative encrypted restore at least quarterly', 'does not set backup schedules or retention')),
    (3, 1, ('Before deviating', 'CTO approves or rejects', 'within 90 days', 'Review monthly',
             'fresh assessment and approval', 'next business day', 'No active exceptions')),
    (3, 3, ('KMS administrative and decrypt audit events monthly and after incidents',
             'those KMS audit logs for at least 12 months', 'access-restricted encrypted storage')),
])
def test_rewrite_retains_scoped_requirements_and_limits(page, section, fragments):
    text = json.loads(PDF.with_suffix('.json').read_text())['pages'][page]['sections'][section]['text']
    assert all(fragment in text for fragment in fragments)



def test_proposed_dates_and_required_actions_do_not_assert_implementation():
    source = json.loads(PDF.with_suffix('.json').read_text(encoding='utf-8'))
    control = source['pages'][0]['sections'][0]['text']
    assert 'Proposed effective date: 2026-10-22; approval not recorded' in control
    assert 'Planned next review: 2027-10-22; approval not recorded' in control
    assert 'Platform Engineering must maintain a certificate inventory' in source['pages'][1]['sections'][1]['text']
    assert 'Platform Engineering must use managed cloud KMS' in source['pages'][2]['sections'][0]['text']
    assert all('\n\n' in s['text'] for s in source['pages'][3]['sections'])
