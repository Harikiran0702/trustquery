"""Offline structural checks; passing tests do not verify semantic labels."""

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import pytest

from trustquery.ingestion.loader import load_pdf, load_text_file


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/eval/evidence_selection_seed.json'
LABELS = {'entailment', 'neutral', 'contradiction'}


def validate(dataset):
    assert set(dataset) == {'schema_version', 'status', 'base_commit', 'cases'}
    assert type(dataset['schema_version']) is int and dataset['schema_version'] == 1
    assert dataset['status'] == 'provisional_requires_user_review'
    assert dataset['base_commit'] == '90fdc1e'
    assert isinstance(dataset['cases'], list) and dataset['cases']
    ids = set()
    pairs = set()
    for case in dataset['cases']:
        assert set(case) == {'id', 'question', 'claim', 'genuine_conflict_exists', 'provenance', 'evidence'}
        for field in ('id', 'question', 'claim'):
            assert isinstance(case[field], str) and case[field].strip()
        assert case['id'] not in ids
        ids.add(case['id'])
        assert type(case['genuine_conflict_exists']) is bool
        cp = case['provenance']
        assert set(cp) == {'question_origin', 'claim_origin', 'annotation_status', 'human_verified', 'requires_user_review'}
        assert cp['question_origin'] and cp['claim_origin']
        assert isinstance(case['evidence'], list) and case['evidence']
        for item in [case, *case['evidence']]:
            p = item['provenance']
            assert p['annotation_status'] == 'provisional_model_authored'
            assert p['human_verified'] is False
            assert p['requires_user_review'] is True
        for e in case['evidence']:
            assert set(e) == {'id', 'text', 'source', 'page', 'expected_relationship', 'relevant_to_claim', 'provenance', 'annotation_rationale'}
            for field in ('id', 'text', 'source', 'annotation_rationale'):
                assert isinstance(e[field], str) and e[field].strip()
            assert e['id'] not in ids
            ids.add(e['id'])
            assert type(e['page']) is int and e['page'] >= 1
            assert e['expected_relationship'] in LABELS
            assert type(e['relevant_to_claim']) is bool
            # Seed convention: only direct support or refutation is relevant.
            assert e['relevant_to_claim'] == (e['expected_relationship'] != 'neutral')
            key = (case['id'], e['text'], e['source'], e['page'])
            assert key not in pairs
            pairs.add(key)
            p = e['provenance']
            assert set(p) == {'kind', 'origin', 'synthetic', 'annotation_status', 'human_verified', 'requires_user_review'}
            assert type(p['synthetic']) is bool
            if p['synthetic']:
                assert p['kind'] == 'synthetic_policy'
                assert p['origin'] == 'authored_for_this_fixture'
                assert e['source'].startswith('SYNTHETIC-')
                assert e['text'].startswith('SYNTHETIC ')
                assert e['page'] == 1  # Virtual page, not a real document.
            else:
                assert p['kind'] == 'local_pdf_snapshot'
                assert p['origin'] == 'data/sample/access_control_policy.pdf'
                assert e['source'] == 'access_control_policy.pdf'
        relationships = {e['expected_relationship'] for e in case['evidence']}
        assert case['genuine_conflict_exists'] == ({'entailment', 'contradiction'} <= relationships)


@pytest.fixture
def seed():
    return json.loads(DATA.read_text(encoding='utf-8'))


def test_seed_schema_and_small_coverage(seed):
    validate(seed)
    assert len(seed['cases']) == 6
    evidence = [e for c in seed['cases'] for e in c['evidence']]
    assert Counter(e['expected_relationship'] for e in evidence) == {
        'entailment': 4, 'neutral': 7, 'contradiction': 3,
    }
    assert sum(c['genuine_conflict_exists'] for c in seed['cases']) == 2
    assert sum(e['provenance']['synthetic'] for e in evidence) == 3
    assert any(all(e['expected_relationship'] == 'neutral' for e in c['evidence']) for c in seed['cases'])
    assert any('contradiction' in {e['expected_relationship'] for e in c['evidence']} and not c['genuine_conflict_exists'] for c in seed['cases'])


def test_local_snapshots_and_origins(seed):
    pages = load_pdf(str(ROOT / 'data/sample/access_control_policy.pdf'))
    q002 = json.loads((ROOT / 'tests/q002_nli_fixture.json').read_text(encoding='utf-8-sig'))
    golden = {q['id']: q for q in json.loads((ROOT / 'data/eval/golden_dataset.json').read_text())}
    assert [p['text'] for p in pages] == [p['text'] for p in q002['pages']]
    # The text document duplicates the quarterly assertion; do not inflate counts.
    assert load_text_file(str(ROOT / 'data/sample/access_control.txt'))['text'] in pages[1]['text']
    for c in seed['cases']:
        qid = c['provenance']['question_origin'].split('#')[1]
        assert c['question'] == golden[qid]['question']
        if qid == 'q002':
            assert c['claim'] == q002['claim']
        for e in c['evidence']:
            if not e['provenance']['synthetic']:
                page = pages[e['page'] - 1]
                assert (e['text'], e['source']) == (page['text'], page['source'])


@pytest.mark.parametrize('defect', [
    'missing_claim', 'invalid_label', 'bad_page', 'duplicate_id',
    'false_human_review', 'missing_review', 'conflict_mismatch',
    'relevance_mismatch', 'unmarked_synthetic', 'duplicate_evidence',
])
def test_schema_rejects_corruption(seed, defect):
    data = deepcopy(seed)
    c = data['cases'][0]
    e = c['evidence'][0]
    if defect == 'missing_claim':
        del c['claim']
    elif defect == 'invalid_label':
        e['expected_relationship'] = 'supported'
    elif defect == 'bad_page':
        e['page'] = True
    elif defect == 'duplicate_id':
        e['id'] = c['id']
    elif defect == 'false_human_review':
        e['provenance']['human_verified'] = True
    elif defect == 'missing_review':
        c['provenance']['requires_user_review'] = False
    elif defect == 'conflict_mismatch':
        c['genuine_conflict_exists'] = True
    elif defect == 'relevance_mismatch':
        e['relevant_to_claim'] = False
    elif defect == 'unmarked_synthetic':
        data['cases'][1]['evidence'][-1]['source'] = 'real_policy.pdf'
    elif defect == 'duplicate_evidence':
        duplicate = deepcopy(e)
        duplicate['id'] = 'different-id'
        c['evidence'].append(duplicate)
    with pytest.raises((AssertionError, KeyError)):
        validate(data)


REVIEW = ROOT / 'data/eval/evidence_selection_review.json'


def validate_review(review, seed):
    assert set(review) == {'schema_version', 'dataset', 'status', 'reviews'}
    assert type(review['schema_version']) is int and review['schema_version'] == 1
    assert review['dataset'] == DATA.name
    assert review['status'] in {'pending_user_approval', 'user_approved'}
    expected = {(c['id'], e['id']): (c, e) for c in seed['cases'] for e in c['evidence']}
    assert isinstance(review['reviews'], list)
    assert len(review['reviews']) == len(expected) == 14
    seen = set()
    for record in review['reviews']:
        assert set(record) == {
            'case_id', 'evidence_id', 'status', 'reviewer', 'reviewed_at',
            'human_verified', 'proposed', 'decision',
        }
        key = (record['case_id'], record['evidence_id'])
        assert key in expected and key not in seen
        seen.add(key)
        assert record['status'] == review['status']
        assert record['reviewed_at'] is None  # Exact approval time is unknown.
        if record['status'] == 'pending_user_approval':
            assert record['reviewer'] is None
            assert record['human_verified'] is False
            assert record['decision'] is None
        else:
            assert record['reviewer'] == 'user (chat approval)'
            assert record['human_verified'] is True
            assert record['decision'] == {
                'relationship': record['proposed']['relationship'],
                'relevant_to_claim': record['proposed']['relevant_to_claim'],
                'approval_context': 'Explicit per-pair user approvals, restated in the continuation request for Project status update (conversation 6ad116c3-4194-83e8-a8f8-729f5fac56ac).',
                'unresolved_fields': ['reviewer_legal_name', 'exact_review_timestamp'],
            }
        case, evidence = expected[key]
        assert record['proposed'] == {
            'claim': case['claim'], 'evidence_text': evidence['text'],
            'source': evidence['source'], 'page': evidence['page'],
            'relationship': evidence['expected_relationship'],
            'relevant_to_claim': evidence['relevant_to_claim'],
            'provenance': evidence['provenance'],
            'rationale': evidence['annotation_rationale'],
            'genuine_conflict_exists': case['genuine_conflict_exists'],
        }
        assert type(record['proposed']['page']) is int
        assert type(record['proposed']['relevant_to_claim']) is bool
        assert type(record['proposed']['genuine_conflict_exists']) is bool
        assert record['proposed']['provenance']['human_verified'] is False
        assert record['proposed']['provenance']['synthetic'] is evidence['provenance']['synthetic']
    assert seen == set(expected)


@pytest.fixture
def review():
    return json.loads(REVIEW.read_text(encoding='utf-8'))


def test_review_coverage_and_linkage(seed, review):
    validate(seed)
    validate_review(review, seed)
    case = next(c for c in seed['cases'] if c['id'] == 'accounts-contradiction-with-neutral')
    assert {e['expected_relationship'] for e in case['evidence']} == {'neutral', 'contradiction'}
    passage = next(c for c in seed['cases'] if c['id'] == 'review-conflict')['evidence'][-1]['text']
    assert 'only once' in passage and 'onlyonce' not in passage
    assert 'accounts-contradiction-only' not in json.dumps(seed)


@pytest.mark.parametrize('defect', [
    'missing_pair', 'duplicate_pair', 'wrong_case', 'unknown_evidence',
    'invalid_status', 'false_approval', 'invented_reviewer', 'invented_date',
    'false_verification', 'fabricated_decision', 'label_drift', 'relevance_drift',
    'source_drift', 'page_drift', 'claim_drift', 'text_drift', 'rationale_drift',
    'synthetic_drift', 'conflict_drift', 'wrong_dataset', 'global_approval',
])
def test_pending_review_rejects_corruption(seed, review, defect):
    data = deepcopy(review)
    data['status'] = 'pending_user_approval'
    for pending in data['reviews']:
        pending.update(status='pending_user_approval', reviewer=None, reviewed_at=None, human_verified=False, decision=None)
    record = data['reviews'][0]
    changes = {
        'wrong_case': ('case_id', 'mfa-conflict'),
        'unknown_evidence': ('evidence_id', 'missing'),
        'invalid_status': ('status', 'unknown'),
        'false_approval': ('status', 'approved'),
        'invented_reviewer': ('reviewer', 'Invented reviewer'),
        'invented_date': ('reviewed_at', '2026-10-21'),
        'false_verification': ('human_verified', True),
        'fabricated_decision': ('decision', {'relationship': 'entailment'}),
    }
    proposed_changes = {
        'label_drift': ('relationship', 'neutral'),
        'relevance_drift': ('relevant_to_claim', False),
        'source_drift': ('source', 'other.pdf'), 'page_drift': ('page', 2),
        'claim_drift': ('claim', 'Changed claim'),
        'text_drift': ('evidence_text', 'Changed passage'),
        'rationale_drift': ('rationale', 'Changed rationale'),
        'conflict_drift': ('genuine_conflict_exists', True),
    }
    if defect == 'missing_pair':
        data['reviews'].pop()
    elif defect == 'duplicate_pair':
        data['reviews'][-1] = deepcopy(record)
    elif defect in changes:
        field, value = changes[defect]
        record[field] = value
    elif defect in proposed_changes:
        field, value = proposed_changes[defect]
        record['proposed'][field] = value
    elif defect == 'synthetic_drift':
        record['proposed']['provenance']['synthetic'] = True
    elif defect == 'wrong_dataset':
        data['dataset'] = 'other.json'
    elif defect == 'global_approval':
        data['status'] = 'approved'
    with pytest.raises((AssertionError, KeyError)):
        validate_review(data, seed)


@pytest.mark.parametrize('defect', [
    'missing_decision', 'wrong_relationship', 'wrong_relevance',
    'missing_context', 'missing_unresolved', 'invented_name', 'invented_time',
    'unverified', 'pending_record', 'case_13_page',
])
def test_approved_review_rejects_corruption(seed, review, defect):
    data = deepcopy(review)
    assert data['status'] == 'user_approved'
    record = data['reviews'][0]
    if defect == 'missing_decision':
        record['decision'] = None
    elif defect == 'wrong_relationship':
        record['decision']['relationship'] = 'neutral'
    elif defect == 'wrong_relevance':
        record['decision']['relevant_to_claim'] = False
    elif defect == 'missing_context':
        del record['decision']['approval_context']
    elif defect == 'missing_unresolved':
        record['decision']['unresolved_fields'] = []
    elif defect == 'invented_name':
        record['reviewer'] = 'Invented reviewer'
    elif defect == 'invented_time':
        record['reviewed_at'] = '2026-10-22T00:00:00Z'
    elif defect == 'unverified':
        record['human_verified'] = False
    elif defect == 'pending_record':
        record['status'] = 'pending_user_approval'
    elif defect == 'case_13_page':
        data['reviews'][12]['proposed']['page'] = 1
    with pytest.raises((AssertionError, KeyError)):
        validate_review(data, seed)
