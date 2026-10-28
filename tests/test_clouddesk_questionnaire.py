"""Offline dataset contract; no production retrieval or generation is invoked."""
import copy
import hashlib
import json
import re
from pathlib import Path

import pytest
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/eval'
LABELS = {'policy-supported', 'policy-only-not-implemented', 'partial', 'unsupported/TBD', 'explicit-negative', 'potentially contradicted'}
DOMAINS = {'access_control', 'authentication', 'encryption_key_management',
           'information_security_governance', 'incident_response',
           'backup_disaster_recovery', 'change_management', 'hr_security', 'vendor_risk'}
POLICIES = {'access_control_policy', 'encryption_key_management_policy',
            'information_security_policy', 'incident_response_plan',
            'backup_disaster_recovery_policy', 'change_management_policy',
            'hr_security_policy', 'vendor_risk_management_policy'}


def normalized(value):
    return re.sub(r'[^a-z0-9]', '', value.lower())


def validate(inputs, key):
    """Reject malformed annotations, leakage and incompatible legacy projections."""
    assert type(inputs) is list and inputs
    assert key['schema_version'] == '1.2'
    assert key['synthetic'] is True
    assert key['provenance'] == 'model_authored'
    assert key['annotation_status'] == 'provisional_model_authored'
    assert key['human_verified'] is False
    assert key['review_status'] == 'requires_user_review'
    assert key['approval_status'] == 'not_recorded'
    assert key['input_file'] == 'clouddesk_questionnaire.json'
    assert re.fullmatch(r'[0-9a-f]{40}', key['policy_snapshot_base_commit'])
    assert key['policy_snapshot_status'] == 'working_tree_manifest_authoritative'
    assert 'policy_snapshot_commit' not in key
    ids = [q['id'] for q in inputs]
    assert len(ids) == len(set(ids))
    assert ids == [f'cdq{i:03}' for i in range(1, len(ids)+1)]
    assert len({normalized(q['question']) for q in inputs}) == len(inputs)
    for q in inputs:
        assert set(q) == {'id', 'question', 'domain', 'response_type'}
        assert q['domain'] in DOMAINS
        assert q['response_type'] in {'yes_no', 'short_answer'}
        assert isinstance(q['question'], str) and len(q['question'].strip()) > 25
    records = key['records']
    assert [r['id'] for r in records] == ids
    assert len({r['id'] for r in records}) == len(ids)
    for r in records:
        assert r['classification'] in LABELS
        assert type(r['should_answer']) is bool
        assert r['annotation_status'] == 'provisional_model_authored'
        assert r['provenance'] == 'model_authored'
        assert r['human_verified'] is False
        assert r['review_status'] == 'requires_user_review'
        assert r['approval_status'] == 'not_recorded'
        assert r['human_review'] == dict(reviewer=None, reviewed_at=None, decision=None, notes=None)
        for field in ('bounded_response', 'expected_evidence_boundary', 'classification_rationale'):
            assert isinstance(r[field], str) and len(r[field].strip()) > 25
        disposition = {'policy-supported': 'answer_as_proposed_policy',
                       'explicit-negative': 'answer_as_source_status'}
        assert r['response_disposition'] == disposition.get(r['classification'], 'bounded_response_and_review')
        assert r['evidence'] and isinstance(r['evidence'], list)
        assert r['policy_families'] == sorted({Path(e['source']).stem for e in r['evidence']})
        if r['classification'] in {'policy-supported', 'explicit-negative'}:
            assert r['should_answer'] is True
            assert r['expected_answer'] == r['bounded_response']
            assert r['expected_source'] == r['evidence'][0]['source']
            assert r['expected_page'] == r['evidence'][0]['page']
        else:
            assert r['should_answer'] is False
            assert r['expected_answer'] is r['expected_source'] is r['expected_page'] is None
        # This snapshot has no genuinely conflicting policies. New conflicts
        # require source review and an explicit revision of this assertion.
        assert r['classification'] != 'potentially contradicted'
        assert r['genuine_conflict_exists'] is False and r['conflict_evidence'] == []
        seen = set()
        for e in r['evidence']:
            assert Path(e['source']).stem in POLICIES
            assert e['source'] == Path(e['source']).name and e['source'].endswith('.pdf')
            assert type(e['page']) is int and e['page'] > 0
            assert all(isinstance(e[f], str) and e[f].strip() for f in ('section', 'page_heading', 'text'))
            assert e['evidence_role'] == 'proposed_requirement_or_scope_limit'
            assert e['expected_evidence_boundary'] == r['expected_evidence_boundary']
            location = (e['source'], e['page'], e['section'])
            assert location not in seen
            seen.add(location)
            assert e['json_source'] == e['source'].replace('.pdf', '.json')
            assert re.fullmatch(r'/pages/\d+/sections/\d+', e['json_pointer'])


@pytest.fixture(scope='module')
def dataset():
    return (json.loads((DATA/'clouddesk_questionnaire.json').read_text(encoding='utf-8')),
            json.loads((DATA/'clouddesk_questionnaire_answer_key.json').read_text(encoding='utf-8')))


@pytest.fixture(scope='module')
def sources():
    sample = ROOT/'data/sample'
    return {stem: ([p.extract_text() for p in PdfReader(sample/f'{stem}.pdf').pages],
                   json.loads((sample/f'{stem}.json').read_text(encoding='utf-8')))
            for stem in POLICIES}


def test_contract_and_coverage(dataset):
    inputs, key = dataset
    validate(inputs, key)
    assert len(inputs) == 100
    assert {q['domain'] for q in inputs} == DOMAINS
    assert {q['response_type'] for q in inputs} == {'yes_no', 'short_answer'}
    assert {r['classification'] for r in key['records']} == LABELS - {'potentially contradicted'}
    assert {Path(e['source']).stem for r in key['records'] for e in r['evidence']} == POLICIES
    assert sum(len(r['policy_families']) > 1 for r in key['records']) >= 30


def test_manifest_pins_exact_source_set(dataset):
    _, key = dataset
    expected = {f'data/sample/{stem}.pdf' for stem in POLICIES}
    expected |= {f'data/sample/{stem}.json' for stem in POLICIES}
    manifest = key['source_manifest']
    assert len(manifest) == len(expected)
    assert {e['path'] for e in manifest} == expected
    for item in manifest:
        assert hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest() == item['sha256']


@pytest.mark.parametrize('stem', sorted(POLICIES))
def test_complete_json_pdf_agreement(stem, sources):
    pages, doc = sources[stem]
    assert doc['synthetic'] is True and doc['review_status'] == 'requires_user_review'
    assert len(pages) == len(doc['pages'])
    for i, page in enumerate(doc['pages']):
        for section in page['sections']:
            assert normalized(section['heading']+section['text']) in normalized(pages[i])


@pytest.mark.parametrize('index', range(100))
def test_each_evidence_locator_and_snapshot(index, dataset, sources):
    _, key = dataset
    for e in key['records'][index]['evidence']:
        pages, doc = sources[Path(e['source']).stem]
        assert e['page'] <= len(pages)
        assert normalized(e['text']) in normalized(pages[e['page']-1])
        if doc:
            parts = e['json_pointer'].split('/')
            page_index, section_index = int(parts[2]), int(parts[4])
            assert page_index + 1 == e['page']
            page = doc['pages'][page_index]
            section = page['sections'][section_index]
            assert (e['page_heading'], e['section'], e['text']) == (page['heading'], section['heading'], section['text'])



@pytest.mark.parametrize('mutation', [
    'leaked_answer', 'duplicate_question_id', 'duplicate_prompt', 'missing_key', 'duplicate_key',
    'label', 'false_human_review', 'approval', 'reviewer', 'empty_boundary',
    'zero_page', 'bool_page', 'wrong_primary', 'unsafe_legacy_answer', 'null_supported',
    'fake_conflict', 'source_path', 'missing_pointer', 'missing_evidence', 'family',
    'global_synthetic', 'format', 'domain', 'rationale',
])
def test_reject_invalid_contract(mutation, dataset):
    inputs, key = copy.deepcopy(dataset)
    r = key['records'][0]
    e = r['evidence'][0]
    unsupported = next(x for x in key['records'] if not x['should_answer'])
    if mutation == 'leaked_answer': inputs[0]['expected_answer'] = r['expected_answer']
    elif mutation == 'duplicate_question_id': inputs[1]['id'] = inputs[0]['id']
    elif mutation == 'duplicate_prompt': inputs[1]['question'] = inputs[0]['question']
    elif mutation == 'missing_key': key['records'].pop()
    elif mutation == 'duplicate_key': key['records'][1]['id'] = r['id']
    elif mutation == 'label': r['classification'] = 'implemented'
    elif mutation == 'false_human_review': r['human_verified'] = True
    elif mutation == 'approval': r['approval_status'] = 'approved'
    elif mutation == 'reviewer': r['human_review']['reviewer'] = 'invented'
    elif mutation == 'empty_boundary': r['expected_evidence_boundary'] = ''
    elif mutation == 'zero_page': e['page'] = 0
    elif mutation == 'bool_page': e['page'] = True
    elif mutation == 'wrong_primary': r['expected_source'] = 'other.pdf'
    elif mutation == 'unsafe_legacy_answer': unsupported['should_answer'] = True
    elif mutation == 'null_supported': r['expected_answer'] = None
    elif mutation == 'fake_conflict': r['genuine_conflict_exists'] = True
    elif mutation == 'source_path': e['source'] = '../access_control_policy.pdf'
    elif mutation == 'missing_pointer': r['evidence'][1]['json_pointer'] = None
    elif mutation == 'missing_evidence': r['evidence'] = []
    elif mutation == 'family': r['policy_families'] = []
    elif mutation == 'global_synthetic': key['synthetic'] = False
    elif mutation == 'format': inputs[0]['response_type'] = 'forced_yes'
    elif mutation == 'domain': inputs[0]['domain'] = 'unknown'
    elif mutation == 'rationale': r['classification_rationale'] = ''
    with pytest.raises((AssertionError, TypeError)):
        validate(inputs, key)


@pytest.mark.parametrize(('question_fragment', 'label', 'response_fragment'), [
    ('What RTO and RPO', 'unsupported/TBD', 'TBD'),
    ('How often is our data backed up', 'unsupported/TBD', 'TBD'),
    ('criminal background checks', 'unsupported/TBD', 'no universal'),
    ('around-the-clock', 'unsupported/TBD', 'No dedicated'),
    ('SOC 2 Type II', 'unsupported/TBD', 'No SOC 2 Type II report is provided'),
    ('How soon would you notify', 'partial', 'No universal deadline'),
    ('Do you force every user', 'unsupported/TBD', 'missing evidence does not establish'),
    ('annual continuity tabletop', 'policy-supported', 'distinct requirements'),
    ('departing employee\'s account appears', 'policy-supported', 'does not authorize delay'),
    ('When keys rotate', 'policy-supported', 'does not re-encrypt'),
    ('second geographic region', 'unsupported/TBD', 'not asserted'),
    ('end-to-end encrypted', 'unsupported/TBD', 'not an assumed capability'),
])
def test_sensitive_boundaries(question_fragment, label, response_fragment, dataset):
    inputs, key = dataset
    q = next(q for q in inputs if question_fragment in q['question'])
    record = next(r for r in key['records'] if r['id'] == q['id'])
    assert record['classification'] == label
    assert response_fragment in record['bounded_response']


def test_legacy_projection_is_loadable_without_changing_corpus(dataset, tmp_path):
    from trustquery.evaluation.dataset import load_golden_dataset
    inputs, key = dataset
    projection = [dict(id=q['id'], question=q['question'], **{
        f:r[f] for f in ('expected_answer', 'expected_source', 'expected_page', 'should_answer')
    }) for q, r in zip(inputs, key['records'], strict=True)]
    path = tmp_path/'projection.json'
    path.write_text(json.dumps(projection), encoding='utf-8')
    assert load_golden_dataset(str(path)) == projection


def test_explicit_negative_is_source_status_not_missing_evidence(dataset):
    _, key = dataset
    records = {r['id']: r for r in key['records']}
    negative = records['cdq031']
    assert negative['classification'] == 'explicit-negative'
    assert negative['should_answer'] is True
    assert negative['response_disposition'] == 'answer_as_source_status'
    assert negative['bounded_response'].startswith('No. The supplied synthetic')
    assert 'has not been approved' in negative['evidence'][0]['text']
    for id_ in ('cdq013', 'cdq035', 'cdq079'):
        assert records[id_]['classification'] == 'unsupported/TBD'
        assert records[id_]['should_answer'] is False
    # Unknown implementation cannot be projected as an answer, nor can an
    # explicit status answer be silently treated as a proposed requirement.
    for field, value in [('should_answer', False),
                         ('response_disposition', 'answer_as_proposed_policy'),
                         ('expected_answer', None)]:
        inputs_copy, key_copy = copy.deepcopy(dataset)
        key_copy['records'][30][field] = value
        with pytest.raises(AssertionError):
            validate(inputs_copy, key_copy)


@pytest.mark.parametrize('id_', ['cdq008', 'cdq028', 'cdq045', 'cdq053',
                                'cdq063', 'cdq088', 'cdq090', 'cdq096'])
def test_direct_polar_questions_use_yes_no(id_, dataset):
    inputs, _ = dataset
    assert next(q for q in inputs if q['id'] == id_)['response_type'] == 'yes_no'


@pytest.mark.parametrize(('id_', 'label', 'fragments'), [
    ('cdq001', 'policy-supported', ('access_control_policy.pdf, page 2', 'hr_security_policy.pdf, page 4')),
    ('cdq008', 'policy-supported', ('under the proposed plan', 'does not authorize delay')),
    ('cdq013', 'unsupported/TBD', ('do not establish whether', 'missing evidence does not establish')),
    ('cdq035', 'unsupported/TBD', ('No SOC 2 Type II report is provided', 'cannot be established')),
    ('cdq039', 'partial', ('do not provide completed evidence', 'dates and limitations')),
    ('cdq041', 'policy-only-not-implemented', ('No completed escalation', 'unverified')),
    ('cdq059', 'partial', ('where justified by risk', 'require validation before adoption')),
    ('cdq061', 'policy-supported', ('proposed recovery policy', 'without guaranteeing')),
    ('cdq079', 'unsupported/TBD', ('no universal background-check requirement', 'No completed screening records')),
    ('cdq080', 'policy-supported', ('before access', 'No signed agreement')),
    ('cdq088', 'policy-supported', ('No identical employee screening', 'does not establish')),
    ('cdq096', 'policy-supported', ('does not require every supplier', 'implementation remains unverified')),
    ('cdq099', 'partial', ('No tested alternate provider', 'remains unverified')),
])
def test_cleanup_preserves_evidence_limits(id_, label, fragments, dataset):
    _, key = dataset
    record = next(r for r in key['records'] if r['id'] == id_)
    assert record['classification'] == label
    assert all(fragment in record['bounded_response'] for fragment in fragments)
    if id_ == 'cdq001':
        assert {(e['source'], e['page']) for e in record['evidence']} == {
            ('access_control_policy.pdf', 2), ('hr_security_policy.pdf', 4)}


def test_review_guide_counts_and_schema_match_dataset(dataset):
    from collections import Counter
    inputs, key = dataset
    guide = (DATA/'clouddesk_questionnaire.md').read_text(encoding='utf-8')
    counts = Counter(r['classification'] for r in key['records'])
    for label in LABELS:
        row = next(line for line in guide.splitlines() if line.startswith(f'| {label} |'))
        assert int(row.split('|')[-2].strip()) == counts[label]
    formats = Counter(q['response_type'] for q in inputs)
    assert f"{formats['short_answer']} short-answer and {formats['yes_no']} Yes/No" in guide
    assert f"There are {sum(r['should_answer'] for r in key['records'])} answerable projections" in guide
    assert f"schema version `{key['schema_version']}`" in guide
    assert f"{sum(len(r['policy_families']) > 1 for r in key['records'])} questions" in guide
