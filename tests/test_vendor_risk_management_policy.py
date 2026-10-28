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
PDF = SAMPLE / 'vendor_risk_management_policy.pdf'


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
        'vendor_policy_builder', ROOT / 'scripts/build_vendor_risk_management_policy.py')
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




def test_governance_alignment():
    source, text = source_text()
    assert len(set(source['related_documents'])) == 7
    for phrase in ['processors/subprocessors', 'concentration', 'geography/jurisdiction',
                   'Engineering Lead reviews', 'before high-risk use',
                   'CTO acceptance of material residual risk', 'critical vendors annually',
                   'after material changes or incidents', 'unique user accounts',
                   'multi-factor authentication for administrative access', 'every quarter',
                   'within 24 hours of termination', 'TLS 1.2 or TLS 1.3', 'AES-256',
                   'managed KMS', 'no default decrypt permission', 'at least every 12 months',
                   'TBD pending BIA/business and contractual review', 'at least quarterly',
                   'at least annually', 'does not replace', 'within 90 days',
                   'reviewed monthly', 'next-business-day CTO review', 'fourth parties',
                   'does not establish that any controls are operating', 'no universal notification deadline']:
        assert phrase in text
    related = {n: ' '.join(p['text'] for p in load_pdf(str(SAMPLE / n)))
               for n in source['related_documents']}
    for name, phrases in {
        'access_control_policy.pdf': ['unique user accounts', 'every quarter', 'within 24 hours'],
        'encryption_key_management_policy.pdf': ['AES-256', 'TLS 1.2 or TLS 1.3', 'no default decrypt permission'],
        'information_security_policy.pdf': ['Reassess critical vendors annually', 'CTO must accept material residual risk'],
        'incident_response_plan.pdf': ['urgent reversible action', 'no universal notification deadline'],
        'backup_disaster_recovery_policy.pdf': ['at least quarterly', 'TBD pending BIA/business and contractual review'],
        'change_management_policy.pdf': ['before enabling access', 'next business day'],
        'hr_security_policy.pdf': ['within 24 hours of termination', 'immediate containment'],
    }.items():
        for phrase in phrases:
            assert phrase in related[name], (name, phrase)


@pytest.fixture(scope='module')
def corpus():
    source, _ = source_text()
    documents = [p for name in [PDF.name] + source['related_documents']
                 for p in load_pdf(str(SAMPLE / name))]
    assert len({p['source'] for p in documents}) == 8
    chunks = chunk_documents(documents)
    for chunk in chunks:
        assert isinstance(chunk['metadata']['chunk_index'], int)
        assert any(chunk['metadata']['source'] == p['source']
                   and chunk['metadata']['page'] == p['page']
                   and chunk['text'] in p['text'] for p in documents)
    return chunks


@pytest.mark.parametrize('query,source,page,answer', [
    ('vendor inventory renewal exit ownership processing locations', PDF.name, 2, 'Inventory and accountable ownership'),
    ('vendor tiering geography concentration substitutability inherent', PDF.name, 2, 'Risk tiering dimensions'),
    ('vendor critical annual reassessment material changes incidents', PDF.name, 2, 'critical vendors annually'),
    ('vendor pre-onboarding privacy due diligence minimization processor', PDF.name, 3, 'Security and privacy due diligence'),
    ('vendor assurance evidence provenance exclusions questionnaires', PDF.name, 3, 'Appropriate evidence'),
    ('vendor contract confidentiality permitted secondary data use subcontractor', PDF.name, 3, 'Contract security'),
    ('vendor approval high-risk missing material evidence', PDF.name, 3, 'blocks planned onboarding'),
    ('vendor continuous monitoring advisories assurance expiry concentration', PDF.name, 4, 'Continuous monitoring proposals'),
    ('vendor fourth parties concentration correlated failures opaque chains', PDF.name, 5, 'Fourth parties'),
    ('vendor exit portability export formats lock-in migration assistance', PDF.name, 6, 'Vendor offboarding and portability'),
    ('vendor data return deletion evidence subcontractors hold limitations', PDF.name, 6, 'Data return, deletion'),
    ('vendor exceptions expired approval monthly deviations', PDF.name, 6, 'Expired approval'),
    ('certificate hostname expiry 30 days', 'encryption_key_management_policy.pdf', 2, '30 days'),
    ('Public Internal Confidential Restricted classification', 'information_security_policy.pdf', 2, 'Restricted'),
    ('SEV1 SEV2 SEV3 SEV4 severity', 'incident_response_plan.pdf', 2, 'SEV1'),
    ('BIA RTO RPO loss tolerance objectives', 'backup_disaster_recovery_policy.pdf', 2, 'BIA'),
    ('standard change catalogue eligibility preauthorization', 'change_management_policy.pdf', 2, 'preauthorization'),
    ('preemployment screening jurisdiction consent role', 'hr_security_policy.pdf', 2, 'no universal background-check requirement'),
])
def test_bm25_across_eight_policies(corpus, query, source, page, answer):
    results = search_bm25(query, corpus, top_k=3)
    assert any(r['metadata']['source'] == source and r['metadata']['page'] == page
               and answer in r['text'] for r in results)


@pytest.mark.parametrize('query,page,limitation', [
    ('actual vendors completed assessments certifications SOC ISO DPA', 1, 'No actual vendors'),
    ('vendor universally required certification SOC ISO suitability', 3, 'No vendor certification is assumed or universally required'),
    ('company-side vendor KMS architecture blindly assign', 4, 'do not blindly assign company KMS architecture'),
    ('existing monitoring tool continuous telemetry coverage', 4, 'No existing monitoring tool'),
    ('vendor advertised RTO RPO backup frequency retention targets', 5, 'TBD pending BIA/business and contractual review'),
    ('vendor annual continuity tabletop quarterly encrypted restore', 5, 'does not replace'),
    ('vendor universal waiting period 24 hours termination', 6, 'not a universal vendor waiting period'),
    ('vendor deletion request alone evidence retained backups', 6, 'rather than declaring deletion from a request alone'),
    ('vendor universal record retention KMS audit logs 12-month', 6, 'not a universal vendor-record retention rule'),
])
def test_abstention_evidence_boundaries(corpus, query, page, limitation):
    # These test retrieval of explicit limitations. They do not execute or certify
    # an answer-generator abstention decision, NLI judgment or human semantic review.
    results = search_bm25(query, corpus, top_k=3)
    assert any(r['metadata']['source'] == PDF.name and r['metadata']['page'] == page
               and limitation in r['text'] for r in results)


def test_unknown_vendor_evidence_requires_abstention(corpus):
    # BM25 returns zero-score matches for absent tokens; never treat them as facts.
    results = search_bm25('ZXQVendorEvidence98765', corpus, top_k=3)
    assert results
    assert all(r['bm25_score'] == 0 for r in results)
    assert all('ZXQVendorEvidence98765' not in r['text'] for r in results)


def test_text_whitespace_and_pdf_binary_temporary_index(tmp_path):
    import os
    import subprocess

    paths = ['data/sample/vendor_risk_management_policy.json',
             'data/sample/vendor_risk_management_policy.pdf',
             'scripts/build_vendor_risk_management_policy.py',
             'tests/test_vendor_risk_management_policy.py']
    for name in [p for p in paths if not p.endswith('.pdf')]:
        raw = (ROOT / name).read_bytes()
        assert raw.endswith(b'\n')
        assert all(line == line.rstrip(b' \t') for line in raw.splitlines())
    original_pdf = PDF.read_bytes()
    real_index = ROOT / '.git/index'
    original_index = real_index.read_bytes()
    objects = tmp_path / 'objects'
    objects.mkdir()
    env = dict(os.environ, GIT_INDEX_FILE=str(tmp_path / 'index'),
               GIT_OBJECT_DIRECTORY=str(objects),
               GIT_ALTERNATE_OBJECT_DIRECTORIES=str(ROOT / '.git/objects'))

    def git(*args):
        return subprocess.run(['git', '-C', str(ROOT), *args], env=env,
                              check=True, capture_output=True).stdout

    attrs = git('check-attr', 'text', 'diff', 'merge', '--', paths[1]).decode()
    for attribute in ['text', 'diff', 'merge']:
        assert f': {attribute}: unset' in attrs

    try:
        git('read-tree', '--empty')
        empty_tree = git('write-tree').decode().strip()

        git('add', '--', *paths)

        stats = git(
            'diff', '--cached', '--numstat', empty_tree
        ).decode()

        assert f'-\t-\t{paths[1]}' in stats
        assert len(stats.splitlines()) == 4
        assert git('show', ':' + paths[1]) == original_pdf
        assert git('diff', '--cached', '--check', empty_tree) == b''

    finally:
        assert real_index.read_bytes() == original_index
        assert PDF.read_bytes() == original_pdf


def test_notification_limit_retrieves_authoritative_incident_policy(corpus):
    # The general notification query ranks existing governing policies above the
    # new vendor contract section; preserve that measured cross-policy result.
    results = search_bm25('vendor breach incident reporting universal notification deadline',
                          corpus, top_k=3)
    assert any(r['metadata']['source'] == 'incident_response_plan.pdf'
               and r['metadata']['page'] == 5
               and 'no universal notification deadline' in r['text'] for r in results)


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


@pytest.mark.parametrize('page,section,requirements', [
    (0, 0, ('dates are proposals', 'not an approved schedule', 'not_recorded', 'requires_user_review')),
    (0, 1, ('fictional 75-person', 'fictional demonstration document', 'does not establish that any controls are operating', 'No actual vendors', 'recorded CTO approval')),
    (0, 2, ('processors/subprocessors', 'trials, free services', 'before enabling access', 'purchasing route')),
    (0, 3, ('business owner and system owner', 'qualified advice', 'No staffed procurement', 'CTO accepts material residual risk', 'All seven related policies')),
    (1, 0, ('Before onboarding', 'legal entity', 'customer data categories', 'support locations', 'evidence limitations', 'retirement')),
    (1, 1, ('geography/jurisdiction', 'cross-border', 'concentration', 'fourth-party', 'inherent risk', 'not evidence of low risk')),
    (1, 2, ('privileged production access', 'sensitive customer data', 'separately from their risk tier', 'Moderate risk', 'Low risk', 'higher applicable tier')),
    (1, 3, ('critical vendors annually', 'after material changes or incidents', 'before changed use', 'at least annually', 'quarterly', 'overdue material risks')),
    (2, 0, ('Before onboarding', 'minimization', 'tenant isolation', 'available logs', 'processor/subprocessor obligations', 'case by case', 'DPA alone')),
    (2, 1, ('if available', 'service coverage', 'exclusions', 'provenance', 'unverified claims', 'No vendor certification', 'SOC 2 report or ISO certificate alone', 'plaintext secrets')),
    (2, 2, ('confidentiality', 'secondary use', 'incident cooperation', 'triggers, recipients, timing basis', 'no universal notification deadline', 'retained backups and holds', 'No executed contract')),
    (2, 3, ('before high-risk use', 'CTO acceptance of material residual risk', 'blocks planned onboarding', 'explicitly accepted', 'supporting review records')),
    (3, 0, ('system-owner approval', 'least privilege', 'every quarter', 'OAuth/API scopes', 'identical employee screening')),
    (3, 1, ('TLS 1.2 or TLS 1.3', 'AES-256', 'managed KMS', 'do not blindly assign company KMS architecture', 'cloud shared responsibility', 'public marketing storage', 'authentication traffic')),
    (3, 2, ('before enabling changed access', 'testing, approval and rollback/recovery', 'no default decrypt permission', 'quarterly', 'at least every 12 months', 'older versions', 'Engineering Lead and CTO approval')),
    (3, 3, ('propose ongoing oversight', 'assurance expiry', 'substitutes', 'No existing monitoring tool', 'does not prove absence of compromise')),
    (4, 0, ('designated alternate', 'preserve evidence', 'urgent reversible containment', 'CTO authorizes material shutdowns', 'external communications', 'earlier notice')),
    (4, 1, ('KMS/key versions', 'TBD pending BIA/business and contractual review', 'agreed SLA', 'tenant boundaries', 'Replication alone', 'retained copies and holds')),
    (4, 2, ('at least quarterly', 'authorized key access', 'at least annually and after material changes', 'common-region failures', 'does not replace', 'No completed restore')),
    (4, 3, ('fourth parties', 'flow-down obligations', 'notice/objection', 'opaque chains', 'correlated failures', 'lock-in', 'own review and incident responsibilities')),
    (5, 0, ('Before critical use', 'manual options', 'export formats', 'dependency validation', 'synthetic data', 'No successful migration', 'OAuth grants', 'verify completion', 'not a universal vendor waiting period', 'immediate containment')),
    (5, 1, ('usable exports', 'subcontractors, retained backups and hold limitations', 'request alone', 'Preserve keys', 'version history', 'subject to holds', 'not a universal vendor-record retention rule', 'No actual audit artifacts')),
    (5, 2, ('Engineering Lead assesses', 'CTO approves or rejects', 'within 90 days', 'reviewed monthly', 'fresh assessment and approval', 'next-business-day CTO review', 'employee termination limits', 'No active exception')),
    (5, 3, ('human reviewers', 'all seven related policies', 'not_recorded', 'requires_user_review', 'does not establish', 'completed vendor assessments', 'implemented security controls', 'obtained independent certification')),
])
def test_rewrite_preserves_vendor_controls_and_limits(page, section, requirements):
    source, _ = source_text()
    text = source['pages'][page]['sections'][section]['text']
    for phrase in requirements:
        assert phrase.lower() in text.lower(), (page + 1, section, phrase)


def test_proposed_dates_and_original_section_locations():
    source, _ = source_text()
    assert 'effective_date' not in source and 'next_review_date' not in source
    assert source['proposed_effective_date'] == '2026-10-22'
    assert source['planned_next_review_date'] == '2027-10-22'
    assert source['date_status'] == 'proposed_dates_approval_not_recorded'
    assert [len(p['sections']) for p in source['pages']] == [4] * 6


def test_external_wording_and_assurance_details():
    source, _ = source_text()
    intro = source['pages'][0]['sections'][1]['text']
    assurance = source['pages'][2]['sections'][1]['text']
    adoption = source['pages'][5]['sections'][3]['text']
    assert 'This proposed policy describes security requirements' in intro
    assert 'model-authored Policy 8 of 8' not in intro
    for phrase in ('security questionnaires', 'architecture explanations', 'relevant policies',
                   'independent assessment summaries if available', 'scoped technical validation',
                   'recovery/deletion descriptions', 'where relevant', 'dates, scope',
                   'service coverage', 'exclusions, provenance and remediation status',
                   'unavailable evidence and unverified claims', 'not establish suitability',
                   'confidential reports', 'plaintext secrets or unnecessary personal data'):
        assert phrase in assurance
    assert max(len(sentence.split()) for sentence in assurance.split('. ')) < 20
    assert all(term not in adoption.lower() for term in ('extraction', 'reproducibility', 'retrieval'))
    assert 'does not establish that CloudDesk has completed vendor assessments' in adoption
