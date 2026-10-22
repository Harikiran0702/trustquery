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
