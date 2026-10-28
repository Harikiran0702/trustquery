"""Active v2 source, provenance, page mapping and deterministic build checks."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from pypdf import PdfReader
from trustquery.ingestion.loader import load_pdf
from trustquery.ingestion.chunker import chunk_documents
from trustquery.retrieval.bm25_retriever import search_bm25

ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / 'data/sample/access_control_policy.pdf'
REQUIREMENTS = [
    ['All employees must use unique user accounts.',
     'Multi-factor authentication is required for administrative access.'],
    ['CloudDesk reviews user access permissions every quarter.',
     'Terminated employee access is revoked within 24 hours.'],
]

def test_requirements_provenance_and_page_locations():
    source = json.loads(DRAFT.with_suffix('.json').read_text())
    assert source['synthetic'] is True
    assert source['provenance'] == 'model_authored'
    assert source['review_status'] == 'requires_user_review'
    assert source['approval_status'] == 'not_recorded'
    assert source['corpus_status'] == 'active_synthetic_evaluation_source_pending_review'
    assert source['owner'] is source['approver'] is source['effective_date'] is None
    assert all(v is None for v in source['human_review'].values())
    pages = load_pdf(str(DRAFT))
    assert len(pages) == len(source['pages']) == 2
    for i, page in enumerate(pages):
        assert page['page'] == i + 1
        for phrase in REQUIREMENTS[i]:
            assert phrase in page['text']
            assert phrase not in pages[1-i]['text']
        for section in source['pages'][i]['sections']:
            assert section['text'] in page['text']
        for marker in ('SYNTHETIC / DEMO', 'model-authored' if i == 0 else 'Model-authored',
                       'requires_user_review', 'not_recorded', f'Page {i+1} of 2'):
            assert marker in page['text']
    assert 'SYNTHETIC DEMO' in PdfReader(DRAFT).metadata.title
    assert 'not_recorded' in PdfReader(DRAFT).metadata.subject
    # The only normative policy content is the original four requirements.
    assert [[s['text'] for s in p['sections'] if s['heading'] in
             ('Employee accounts', 'Administrative access', 'Access reviews', 'Terminated employees')]
            for p in source['pages']] == REQUIREMENTS

def test_migration_records_legacy_hash_without_claiming_current_bytes():
    source = json.loads(DRAFT.with_suffix('.json').read_text())
    legacy = source['legacy_source']
    assert legacy['sha256'] == '1e570731e788d3449a1e1e181a6d04927d011d87add4ce78becd52a94efb128b'
    assert legacy['path'] == 'data/sample/access_control_policy.pdf'
    assert hashlib.sha256(DRAFT.read_bytes()).hexdigest() != legacy['sha256']

def test_reproducible_build(tmp_path):
    pytest.importorskip('reportlab', reason='Optional PDF authoring tool')
    spec = importlib.util.spec_from_file_location('access_builder', ROOT / 'scripts/build_access_control_policy.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    first = builder.build(output_path=tmp_path / 'one.pdf').read_bytes()
    second = builder.build(output_path=tmp_path / 'two.pdf').read_bytes()
    assert first == second == DRAFT.read_bytes()

@pytest.mark.parametrize('query,page,phrase', [
    ('unique user accounts administrative access', 1, 'unique user accounts'),
    ('user access permissions every quarter', 2, 'every quarter'),
    ('terminated employee access 24 hours', 2, 'within 24 hours'),
])
def test_draft_retrieval(query, page, phrase):
    documents = load_pdf(str(DRAFT))
    for pdf in sorted((ROOT / 'data/sample').glob('*.pdf')):
        if pdf.name != DRAFT.name:
            documents.extend(load_pdf(str(pdf)))
    results = search_bm25(query, chunk_documents(documents), top_k=5)
    assert any(r['metadata']['source'] == DRAFT.name
               and r['metadata']['page'] == page and phrase in r['text']
               for r in results)
