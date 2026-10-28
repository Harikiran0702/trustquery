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
PDF = SAMPLE / 'change_management_policy.pdf'


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
            assert ' '.join(section['text'].split()) in actual['text']
    metadata = PdfReader(PDF).metadata
    assert 'SYNTHETIC DEMO' in metadata.title
    assert 'fictional' in metadata.author
    assert 'requires_user_review' in metadata.subject


def get_builder():
    pytest.importorskip('reportlab', reason='Optional PDF authoring tool')
    spec = importlib.util.spec_from_file_location(
        'incident_plan_builder', ROOT / 'scripts/build_change_management_policy.py')
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


def test_authority_alignment_and_limits():
    source, text = source_text()
    assert len(set(source['related_documents'])) == 5
    for phrase in ['at least annually', 'every quarter', 'within 24 hours',
        'AES-256', 'TLS 1.2 or TLS 1.3', 'at least every 12 months',
        'at least quarterly', 'within 90 days', 'reviewed monthly',
        'minimum 12-month retention for KMS audit logs only',
        'TBD pending BIA/business and contractual review',
        'Engineering Lead and CTO approval', 'next-business-day review',
        'not operating facts', 'No implemented controls']:
        assert phrase in text
    related = {name: ' '.join(p['text'] for p in load_pdf(str(SAMPLE / name)))
               for name in source['related_documents']}
    for name, phrases in {
        'access_control_policy.pdf': ['unique user accounts', 'every quarter', 'within 24 hours'],
        'encryption_key_management_policy.pdf': ['AES-256', 'at least every 12 months',
            'Engineering Lead and CTO approval', 'no default decrypt permission'],
        'information_security_policy.pdf': ['at least annually, before material service changes',
            'Review open risks quarterly'],
        'incident_response_plan.pdf': ['urgent reversible action', 'Engineering Lead verifies containment'],
        'backup_disaster_recovery_policy.pdf': ['at least quarterly', 'TBD pending BIA/business and contractual review',
            'does not replace', 'minimum 12-month retention for KMS audit logs'],
    }.items():
        for phrase in phrases:
            assert phrase in related[name], (name, phrase)


@pytest.mark.parametrize('heading,required,forbidden', [
    ('Standard changes - bounded preauthorization',
     ['Each execution still needs a ticket', 'requires normal or emergency handling'],
     ['all routine changes are automatically approved']),
    ('Normal changes and approval gates',
     ['Before deployment require peer review', 'not by itself proof'], ['self-approval is sufficient']),
    ('Tenant isolation and migration tests',
     ['negative cross-tenant access attempts', 'block planned release'], ['ignore isolation failures']),
    ('Immediate containment authority',
     ['Do not delay immediate containment', 'urgent reversible action', 'verified alternate approval'],
     ['wait for CTO approval before stopping active harm']),
    ('Emergency safeguards and next business day review',
     ['CTO reviews emergency changes by the next business day', 'omitted checks', 'remaining risk'],
     ['emergency changes need no review']),
    ('Failed changes, rollback and forward recovery',
     ['only if safe', 'irreversible migrations', 'Link or open an incident'], ['always roll back immediately']),
    ('Exceptions expire and require review',
     ['within 90 days', 'reviewed monthly', 'Expired approval cannot authorize'], ['permanent exception']),
    ('Proposed audit trail and retention',
     ['not actual audit artifacts', 'not a universal'], ['all change records retained for 12 months']),
])
def test_negative_scenarios_and_nonclaims(heading, required, forbidden):
    source, _ = source_text()
    text = next(s['text'] for p in source['pages'] for s in p['sections'] if s['heading'] == heading)
    for phrase in required:
        assert phrase in text
    for phrase in forbidden:
        assert phrase not in text


@pytest.fixture(scope='module')
def corpus():
    policy, _ = source_text()
    documents = [d for name in [PDF.name] + policy['related_documents']
                 for d in load_pdf(str(SAMPLE / name))]
    chunks = chunk_documents(documents)
    assert len({d['source'] for d in documents}) == 6
    for chunk in chunks:
        original = next(d for d in documents if d['source'] == chunk['metadata']['source']
                        and d['page'] == chunk['metadata']['page'])
        assert chunk['text'] in original['text']
        assert isinstance(chunk['metadata']['chunk_index'], int)
    return chunks


@pytest.mark.parametrize('query,source,page,answer', [
    ('standard change catalogue eligibility preauthorization', PDF.name, 2, 'preauthorization'),
    ('change ticket impact privacy timing rollback', PDF.name, 2, 'Ticketed change request'),
    ('reviewed commit immutable release artifact CI pull requests', PDF.name, 3, 'artifact'),
    ('negative cross-tenant access migration tests', PDF.name, 3, 'negative cross-tenant access attempts'),
    ('Failed tenant isolation authorization integrity block planned release', PDF.name, 3, 'block planned release'),
    ('change freeze calendar defer planned', PDF.name, 4, 'change freeze'),
    ('canary cohorts feature flags rollout', PDF.name, 4, 'canary'),
    ('failed change irreversible migrations forward recovery', PDF.name, 4, 'only if safe'),
    ('emergency omitted checks next business day CTO reviews', PDF.name, 5, 'next business day'),
    ('change exceptions expired approval monthly renewal', PDF.name, 6, 'Expired approval'),
    ('unique user accounts administrative access', 'access_control_policy.pdf', 1, 'unique user accounts'),
    ('certificate hostname expiry 30 days', 'encryption_key_management_policy.pdf', 2, '30 days'),
    ('Public Internal Confidential Restricted classification', 'information_security_policy.pdf', 2, 'Restricted'),
    ('SEV1 SEV2 SEV3 SEV4 severity', 'incident_response_plan.pdf', 2, 'SEV1'),
    ('BIA RTO RPO loss tolerance objectives', 'backup_disaster_recovery_policy.pdf', 2, 'BIA'),
])
def test_bm25_across_six_policies(corpus, query, source, page, answer):
    results = search_bm25(query, corpus, top_k=3)
    assert any(r['metadata']['source'] == source and r['metadata']['page'] == page
               and answer in r['text'] for r in results)


@pytest.mark.parametrize('query,page,limitation', [
    ('standard catalogue supplied approved preauthorization', 2, 'No catalogue is supplied or approved here'),
    ('emergency containment delay routine approval', 5, 'Do not delay immediate containment'),
    ('backup frequency retention RTO RPO numerical', 5, 'sets no numerical recovery objective'),
    ('retention KMS change-ticket universal', 6, 'not a universal'),
    ('certification SOC 2 SLA implemented controls', 1, 'No implemented controls'),
])
def test_retrieved_limitations_do_not_support_operational_claims(corpus, query, page, limitation):
    # Offline lexical retrieval checks evidence boundaries, not an NLI or compliance judgment.
    results = search_bm25(query, corpus, top_k=3)
    assert any(r['metadata']['source'] == PDF.name and r['metadata']['page'] == page
               and limitation in r['text'] for r in results)


@pytest.mark.parametrize('page,section,requirements', [
    (0, 2, ('database schemas and data migrations', 'infrastructure as code (IaC)',
            'manual console changes, feature flags, scripts and vendor-driven changes',
            'development work with a production impact', 'remain authoritative')),
    (0, 3, ('competent independent reviewer', 'Separate requester, reviewer and deployment approval duties',
            'CTO approval of compensating review before a planned deployment', 'do not silently self-approve')),
    (1, 0, ('documented, tested procedure preauthorized', 'CTO must accept any material residual risk',
            'eligibility, exact scope, version, validation, rollback, authorized operators',
            'Each execution still needs a ticket', 'requires normal or emergency handling')),
    (1, 1, ('peer review, system-owner acceptance', 'Engineering Lead process approval',
            'CTO approves material residual risk, major service impact and exceptions',
            'Reassess and reapprove', 'not by itself proof')),
    (1, 2, ('unique ID', 'requester, owner, classification rationale', 'tenant scope and dependency map',
            'exact artifact/configuration versions', 'data migration reversibility',
            'proposed timing and time zone', 'verification/stop criteria', 'decision authority')),
    (2, 0, ('exact reviewed commit and immutable release artifact identifier or digest',
            'unit, integration and migration checks', 'Block planned release', 'approved, bounded exception')),
    (2, 2, ('negative cross-tenant access attempts', 'object identifiers, API authorization, exports',
            'background jobs, caches and privilege changes', 'old/new application coexistence',
            'duplicate event handling', 'partially applied migrations', 'block planned release')),
    (2, 3, ('representative encrypted restore', 'authorized key access', 'before planned rollout',
            'backup-specific key grants', 'required older key versions', 'holds',
            'at least every 12 months', 'does not re-encrypt historical data',
            'Engineering Lead and CTO approval', 'at least quarterly', 'at least annually',
            'neither substitutes for the other')),
    (3, 1, ('approvals or valid standard preauthorization', 'reviewed artifact identity',
            'recovery readiness', 'canary cohorts or feature flags', 'reconcile them into reviewed configuration/IaC',
            'Pause when monitoring or responsible authorized coverage is unavailable')),
    (3, 2, ('error rates, latency, security signals, tenant boundaries, data integrity',
            'acceptance criteria', 'observation period', 'expand, hold or stop',
            'system owner records acceptance', 'CTO approves material residual risk')),
    (3, 3, ('Stop rollout', 'only if safe', 'irreversible migrations', 'decision authority before planned deployment',
            'Preserve evidence', 'Link or open an incident', 'Retest before redeployment')),
    (4, 0, ('urgent reversible action', 'Do not delay immediate containment',
            'as soon as safe', 'CTO authorizes material service shutdowns', 'verified alternate approval',
            'does not expand authority for destructive actions, emergency decrypt access or external communications')),
    (4, 1, ('omitted checks', 'known uncertainty', 'CTO reviews emergency changes by the next business day',
            'remaining risk', 'reconcile hotfixes into version control', 'prior approval')),
    (4, 2, ('revoke affected grants', 'disable compromised versions where safe',
            'Never blindly destroy keys', 'temporary emergency decrypt access', 'next-business-day review',
            'destruction approvals', 'clean recovery, tenant integrity and dependency gates',
            'TBD pending BIA/business and contractual review')),
    (4, 3, ('OAuth scopes, data flows, subprocessors, regions, exports and deletion behavior',
            'before enabling access', 'Engineering Lead reviews', 'CTO accepts material residual risk',
            'critical vendors annually and after material changes or incidents', 'exit/recovery limitations')),
    (5, 0, ('CI/test outputs', 'approval decisions and timestamps', 'deployment actor/actions',
            'production verification, rollback, incident linkage and closure', 'version history',
            'approved retention and holds', 'minimum 12-month retention for KMS audit logs only',
            'not a universal change-ticket, incident-evidence or backup retention rule')),
    (5, 1, ('compensating safeguards, remediation owner and expiry', 'CTO approves or rejects',
            'within 90 days', 'reviewed monthly', 'fresh assessment and approval',
            'Expired approval cannot authorize', 'CTO review by the next business day')),
    (5, 2, ('at least quarterly', 'significant failed changes or incidents',
            'Suspend unsafe standard procedures', 'retest before reuse', 'CTO risk acceptance', 'at least annually')),
])
def test_wording_preserves_change_safety_and_authority(page, section, requirements):
    source, _ = source_text()
    text = source['pages'][page]['sections'][section]['text']
    for phrase in requirements:
        assert phrase in text, (page + 1, section, phrase)


def test_wording_mapping_and_unapproved_dates():
    source, _ = source_text()
    assert [len(p['sections']) for p in source['pages']] == [4] * 6
    assert 'effective_date' not in source and 'next_review_date' not in source
    assert source['proposed_effective_date'] == '2026-10-22'
    assert source['planned_next_review_date'] == '2027-10-22'
    assert source['date_status'] == 'proposed_dates_approval_not_recorded'
    assert 'not an approved schedule' in source['pages'][0]['sections'][0]['text']


def test_final_semantic_review_refinements():
    source, _ = source_text()
    scope = source['pages'][0]['sections'][2]['text']
    for requirement in ('application code', 'database schemas', 'data migrations',
                        'infrastructure as code (IaC)', 'cloud configuration',
                        'security controls', 'identity permissions', 'encryption keys',
                        'backup methods', 'recovery runbooks', 'vendor integrations',
                        'manual console changes', 'feature flags', 'scripts',
                        'vendor-driven changes', 'development work with a production impact'):
        assert requirement.lower() in scope.lower()
    safeguards = source['pages'][2]['sections'][0]['text']
    for requirement in ('pull request', 'exact reviewed commit', 'immutable release artifact',
                        'peer review', 'CI/tests', 'Keep test evidence', 'Block planned release',
                        'approved, bounded exception'):
        assert requirement in safeguards
    assert source['pages'][5]['sections'][3]['text'].endswith(
        'This proposed policy does not provide evidence that CloudDesk has implemented or tested these controls.')
