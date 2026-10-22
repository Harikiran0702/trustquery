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
PDF = SAMPLE / 'hr_security_policy.pdf'


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
        assert f'Page {number} of 6' in actual['text']
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
        'hr_policy_builder', ROOT / 'scripts/build_hr_security_policy.py')
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



def test_status_and_cross_policy_termination_linkage():
    source, text = source_text()
    assert len(source['related_documents']) == 6
    for phrase in ['not operating facts', 'No established HR department',
                   'no actual training delivery', 'no universal background-check requirement',
                   'no fixed screening frequency', 'unique user accounts',
                   'multi-factor authentication for administrative access',
                   'at onboarding and annually', 'every quarter',
                   'within 24 hours of termination', 'immediate containment',
                   'TBD pending applicable laws/contracts', 'incident/evidence holds',
                   'expire within 90 days', 'reviewed monthly', 'next-business-day CTO review',
                   'at least annually', 'no default decrypt permission']:
        assert phrase in text
    access = ' '.join(p['text'] for p in load_pdf(str(SAMPLE / 'access_control_policy.pdf')))
    assert 'within 24 hours' in access
    assert 'every quarter' in access
    assert 'unique user accounts' in access
    assert source['approval_status'] == 'not_recorded'


@pytest.fixture(scope='module')
def corpus():
    source, _ = source_text()
    documents = [p for name in [PDF.name] + source['related_documents']
                 for p in load_pdf(str(SAMPLE / name))]
    assert len({p['source'] for p in documents}) == 7
    chunks = chunk_documents(documents)
    for chunk in chunks:
        assert any(chunk['metadata']['source'] == p['source']
                   and chunk['metadata']['page'] == p['page']
                   and chunk['text'] in p['text'] for p in documents)
    return chunks


@pytest.mark.parametrize('query,source,page,answer', [
    ('preemployment screening jurisdiction consent role', PDF.name, 2, 'no universal background-check requirement'),
    ('confidentiality NDAs acceptable use acknowledgments', PDF.name, 2, 'Confidentiality and acceptable use'),
    ('onboarding approvals provisioned identity start date sponsor', PDF.name, 3, 'unique user accounts'),
    ('privileged role preparation training decrypt permission', PDF.name, 3, 'no default decrypt permission'),
    ('training records attendance completion overdue', PDF.name, 3, 'Training records'),
    ('transfers leave return dates obsolete permissions', PDF.name, 4, 'reassess on return'),
    ('terminated employee access 24 hours termination', PDF.name, 4, 'within 24 hours'),
    ('offboarding asset return sessions tokens holds', PDF.name, 4, 'Returning a device alone'),
    ('insider threat investigation privacy impartial investigator', PDF.name, 5, 'No blanket employee surveillance'),
    ('personnel retention approved laws contracts', PDF.name, 5, 'TBD pending applicable laws/contracts'),
    ('exceptions expire monthly emergency CTO', PDF.name, 6, 'within 90 days'),
    ('certificate hostname expiry 30 days', 'encryption_key_management_policy.pdf', 2, '30 days'),
    ('Public Internal Confidential Restricted classification', 'information_security_policy.pdf', 2, 'Restricted'),
    ('SEV1 SEV2 SEV3 SEV4 severity', 'incident_response_plan.pdf', 2, 'SEV1'),
    ('BIA RTO RPO loss tolerance objectives', 'backup_disaster_recovery_policy.pdf', 2, 'BIA'),
    ('standard change catalogue eligibility preauthorization', 'change_management_policy.pdf', 2, 'preauthorization'),
    ('user access permissions reviewed quarter', 'access_control_policy.pdf', 2, 'quarter'),
])
def test_bm25_across_seven_policies(corpus, query, source, page, answer):
    results = search_bm25(query, corpus, top_k=3)
    assert any(r['metadata']['source'] == source and r['metadata']['page'] == page
               and answer in r['text'] for r in results)


@pytest.mark.parametrize('query,page,limitation', [
    ('universal background check fixed screening frequency', 2, 'no universal background-check requirement'),
    ('actual training delivery completion competence', 3, 'no actual training delivery'),
    ('24 hour termination waiting period urgent compromise', 4, 'do not delay containment'),
    ('personnel universal retention period', 5, 'No universal personnel retention period'),
    ('blanket surveillance private accounts guilt', 5, 'No blanket employee surveillance'),
    ('automatic sanction employment outcome', 5, 'no automatic sanction'),
    ('exception unlawful screening 24 hour termination', 6, 'cannot authorize unlawful screening'),
    ('established HR department completed screening certification', 1, 'No established HR department'),
])
def test_retrieved_limits_reject_operational_claims(corpus, query, page, limitation):
    # Lexical evidence boundaries, not an NLI or human semantic approval test.
    results = search_bm25(query, corpus, top_k=3)
    assert any(r['metadata']['source'] == PDF.name and r['metadata']['page'] == page
               and limitation in r['text'] for r in results)
