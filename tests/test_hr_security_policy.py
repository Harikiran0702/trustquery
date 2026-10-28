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
    ('offboarding asset return sessions tokens holds', PDF.name, 4, 'revoke sessions, tokens'),
    ('insider threat investigation privacy impartial investigator', PDF.name, 5, 'No blanket employee surveillance'),
    ('personnel retention approved laws contracts', PDF.name, 5, 'TBD pending applicable laws/contracts'),
    ('certificate hostname expiry 30 days', 'encryption_key_management_policy.pdf', 2, '30 days'),
    ('Public Internal Confidential Restricted classification', 'information_security_policy.pdf', 2, 'Restricted'),
    ('SEV1 SEV2 SEV3 SEV4 severity', 'incident_response_plan.pdf', 2, 'SEV1'),
    ('BIA RTO RPO loss tolerance objectives', 'backup_disaster_recovery_policy.pdf', 2, 'BIA'),
    ('standard change catalogue eligibility preauthorization', 'change_management_policy.pdf', 2, 'preauthorization'),
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


def test_quarterly_access_review_evidence_and_governing_source(corpus):
    # BM25 uses whitespace tokens: reviewed != reviews; quarter differs from quarter with a period.
    # A broad query may retrieve the same obligation in another policy first.
    query = 'user access permissions reviewed quarter'
    results = search_bm25(query, corpus, top_k=3)
    assert any(r['metadata']['source'] in {
        'access_control_policy.pdf', 'encryption_key_management_policy.pdf',
        'hr_security_policy.pdf', 'backup_disaster_recovery_policy.pdf',
        'information_security_policy.pdf'}
        and 'user access permissions every quarter' in r['text'] for r in results)
    # Require retrieval of the governing page with a wording-aligned query as
    # well as checking its corpus content; do not increase the retrieval budget.
    direct = search_bm25(
        'CloudDesk reviews user access permissions every quarter.', corpus, top_k=3)
    assert any(r['metadata']['source'] == 'access_control_policy.pdf'
               and r['metadata']['page'] == 2
               and 'CloudDesk reviews user access permissions every quarter.' in r['text']
               for r in direct)
    # Independently verify the governing source still contains the requirement.
    governing = [c for c in corpus
                 if c['metadata']['source'] == 'access_control_policy.pdf']
    assert any(c['metadata']['page'] == 2
               and 'CloudDesk reviews user access permissions every quarter.' in c['text']
               for c in governing)


def test_exception_retrieval_general_and_hr_specific(corpus):
    # Shared exception language can rank another governing policy first.
    # Preserve the broad query/budget, then require the HR-specific restriction.
    results = search_bm25('exceptions expire monthly emergency CTO', corpus, top_k=3)
    assert any('within 90 days' in r['text'] and
               ('monthly' in r['text']) and 'CTO' in r['text'] for r in results)
    direct = search_bm25('exception unlawful screening 24 hour termination', corpus, top_k=3)
    assert any(r['metadata']['source'] == PDF.name and r['metadata']['page'] == 6
               and 'within 90 days' in r['text'] and 'reviewed monthly' in r['text']
               and 'cannot authorize unlawful screening' in r['text'] for r in direct)


def test_offboarding_device_return_does_not_replace_revocation(corpus):
    # The broad query retrieves the revocation actions; this distinct question
    # targets the device-return limitation beyond that chunk's boundary.
    results = search_bm25('offboarding Returning a device alone does not revoke access',
                          corpus, top_k=3)
    assert any(r['metadata']['source'] == PDF.name and r['metadata']['page'] == 4
               and 'Returning a device alone does not revoke access.' in r['text']
               for r in results)


@pytest.mark.parametrize('page,section,requirements', [
    (0, 1, ('fictional 75-person B2B SaaS', 'model-authored', 'not operating facts',
            'No established HR department', 'completed screening', 'recorded CTO approval')),
    (0, 2, ('employees, temporary personnel, contractors and vendor workers',
            'System owners approve business access', 'CTO accepts material residual risk')),
    (1, 0, ('role, access sensitivity and risk', 'lawful authority', 'required consent',
            'no universal background-check requirement', 'no fixed screening frequency')),
    (1, 1, ('relevant, verified information', 'disputed findings', 'minimal personal information',
            'Approve retention before collection', 'no completed screening')),
    (1, 2, ('Before granting access', 'confidentiality agreements or NDAs',
            'acceptable-use acknowledgments', 'continuing confidentiality', 'No signed agreement')),
    (1, 3, ('Before access', 'sponsor', 'start/end dates', 'information return/deletion',
            'identical employee screening', 'critical vendors annually')),
    (2, 0, ('System owners approve least privilege before', 'unique user accounts',
            'multi-factor authentication for administrative access', 'before access is enabled')),
    (2, 1, ('Before privileged duties', 'Engineering Lead approves key grants',
            'no default decrypt permission', 'Do not silently self-approve')),
    (2, 2, ('at onboarding and annually', 'accessible', 'no actual training delivery')),
    (2, 3, ('attendance/completion', 'temporary access restrictions', 'no universal training deadline')),
    (3, 0, ('promptly notify', 'system-owner approval', 'remove obsolete permissions',
            'Record approval before reactivation', 'approved holds')),
    (3, 1, ('every quarter', 'Managers confirm', 'System owners validate', 'CTO')),
    (3, 2, ('within 24 hours of termination at the latest', 'not a waiting period',
            'revoke affected access immediately', 'do not delay containment',
            'urgent reversible action', 'CTO retains material shutdown', 'contractor/vendor')),
    (3, 3, ('asset return', 'revoke sessions, tokens', 'verified revocation times',
            'without sharing credentials', 'incident/evidence holds', 'does not authorize key destruction')),
    (4, 0, ('necessary purpose', 'consent where required', 'need to know',
            'TBD pending applicable laws/contracts', 'before collection or disposal')),
    (4, 1, ('promptly report', 'unavailable or implicated', 'No live reporting service',
            'no automatic sanction', 'avoid retaliation')),
    (4, 2, ('documented authority', 'impartial authorized investigator', 'privacy/legal advice',
            'No blanket employee surveillance', 'fair opportunity')),
    (4, 3, ('collection provenance', 'chain-of-custody', 'Preserve originals',
            'incident/evidence holds', 'KMS audit logs only', 'not a personnel-record retention rule')),
    (5, 0, ('Engineering Lead assesses', 'CTO approves or rejects', 'within 90 days',
            'reviewed monthly', 'fresh assessment and approval', 'next-business-day CTO review',
            'cannot authorize unlawful screening', 'original 24-hour termination requirement')),
    (5, 1, ('proposed evidence types', 'not actual personnel files', 'Minimize sensitive contents')),
    (5, 2, ('at least annually', 'material role, vendor or service changes', 'quarterly access-review')),
    (5, 3, ('human reviewers', 'CTO approval', 'not_recorded', 'requires_user_review',
            'does not provide evidence that CloudDesk has implemented or tested')),
])
def test_rewrite_preserves_personnel_safety_boundaries(page, section, requirements):
    source, _ = source_text()
    text = source['pages'][page]['sections'][section]['text']
    for phrase in requirements:
        assert phrase.lower() in text.lower(), (page + 1, section, phrase)


def test_proposed_dates_and_section_mapping():
    source, _ = source_text()
    assert [len(p['sections']) for p in source['pages']] == [4] * 6
    assert 'effective_date' not in source and 'next_review_date' not in source
    assert source['proposed_effective_date'] == '2026-10-22'
    assert source['planned_next_review_date'] == '2027-10-22'
    assert source['date_status'] == 'proposed_dates_approval_not_recorded'


def test_final_editorial_refinements_preserve_evidence_categories():
    source, _ = source_text()
    scope = source['pages'][0]['sections'][2]['text']
    assert 'Managers or assigned personnel coordinators are responsible for arranging onboarding, role changes, and employee departures.' in scope
    assert 'CloudDesk does not assume a dedicated HR department' in scope
    onboarding = source['pages'][2]['sections'][0]['text']
    for phrase in ('verifies identity', 'System owners approve least privilege before provisioning',
                   'Authorized operators', 'unique user accounts',
                   'multi-factor authentication for administrative access',
                   'requester, approver, operator, permissions and completion time',
                   'approval and provisioning', 'Do not copy a predecessor account'):
        assert phrase in onboarding
    records = source['pages'][5]['sections'][1]['text']
    for phrase in ('Proposed personnel records', 'Proposed access and security evidence',
                   'role risk assessments', 'lawful screening and consent decisions',
                   'confidentiality agreements', 'acceptable-use acknowledgments',
                   'training assignments, completion and follow-up', 'onboarding access approvals',
                   'identity/MFA checks', 'transfer/leave tickets', 'quarterly reviews',
                   'offboarding revocation times and asset returns', 'contractor sponsor/end-date reviews',
                   'authorized investigation records', 'retention/hold decisions',
                   'exceptions and policy review feedback', 'not actual personnel files',
                   'Minimize sensitive contents', 'restricted custodians'):
        assert phrase in records
