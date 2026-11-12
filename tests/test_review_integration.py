"""Real-batch integration: copied input, disposable SQLite, independent XLSX reader."""
from collections import Counter
from io import BytesIO
import json
from pathlib import Path
import shutil
import pytest
from openpyxl import load_workbook
from trustquery.review_store import ReviewStore, ReviewConflict, load_batch, policy_extract
from trustquery.review_export import export_bytes, main

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture
def real_batch(tmp_path):
    source = ROOT/'data/batch_runs/major_deliverable_1/results.json'
    batch = tmp_path/'data/batch_runs/major_deliverable_1/results.json'
    batch.parent.mkdir(parents=True)
    shutil.copyfile(source, batch)
    rows, digest = load_batch(batch)
    assert len(rows) == 100
    db = tmp_path/'data/reviews/major_deliverable_2.sqlite3'
    return batch, db, rows, digest

def save(store, r, action, revision=0, answer=None):
    selected = [r['evidence_references'][0]['chunk_id']]
    return store.save(r, revision=revision, action=action,
        answer=policy_extract(r, selected) if answer is None else answer,
        selected=selected, reviewer='=REVIEWER()', note='@Missing proof / reviewed bounded scope', scope_checked=True)

def reconcile(batch, db, rows, digest):
    store = ReviewStore(db, digest)
    before = db.read_bytes()
    content = export_bytes(batch, db)
    assert content == export_bytes(batch, db)
    book = load_workbook(BytesIO(content))
    def table(name):
        data = list(book[name].values)
        return [dict(zip(data[0], ('' if v is None else v for v in row))) for row in data[1:]]
    questions = table('Questions')
    assert len(questions) == len(rows) == 100
    assert table('Verified citations') == []
    candidates = table('UNVERIFIED candidates')
    expected = []
    for r, q in zip(rows, questions):
        s = store.current(r)
        saved = s['revision'] > 0
        assert q['question_id'] == r['id']
        assert q['question_text'] == r['question']
        assert q['review_status'] == q['stored_status'] == s['status']
        assert q['revision'] == str(s['revision'])
        assert q['reviewed_answer'] == (s['answer'] if saved else '')
        assert q['approved_policy_extract'] == (s['answer'] if s['status']=='approved' else '')
        assert q['reviewer'] == s['reviewer']
        assert q['reviewer_decision_note'] == s['note']
        assert q['scope_checked'] == str(s['scope_checked']).lower()
        assert q['approval_scope'] == ('synthetic_policy_extract_only' if s['status']=='approved' else '')
        assert q['outcome'] == r['outcome']
        assert q['batch_answer_unreviewed'] == r['answer']
        assert q['batch_sha256'] == digest and q['operational_proof'] == 'false'
        missing = r['evidence_assessment'].get('implementation_required') is not False or r['outcome']=='implementation_evidence_required' or s['status']=='needs_evidence'
        assert q['implementation_evidence_missing'] == str(missing).lower()
        assert json.loads(q['selected_chunk_ids_json']) == s['selected']
        assert json.loads(q['provenance_json']) == r.get('provenance', {})
        assert json.loads(q['assessment_json']) == r['evidence_assessment']
        assert json.loads(q['errors_json']) == r.get('errors', [])
        assert json.loads(q['stored_citations_unverified_json']) == s['citations'] == []
        assert json.loads(q['batch_citations_unverified_json']) == r.get('citations', [])
        assert json.loads(q['saved_review_json']) == (s if saved else {})
        if saved:
            assert q['saved_at_utc'] == store.history(r['id'])[-1]['timestamp']
        else:
            assert q['saved_at_utc'] == ''
        for e in r['evidence_references']:
            expected.append(dict(question_id=r['id'], evidence_status='UNVERIFIED CANDIDATE',
                selected_for_review=str(e['chunk_id'] in s['selected']).lower(), source=e['source'],
                page_1_based=str(e['page']), chunk_index_0_based=str(e['chunk_index']),
                chunk_id=e['chunk_id'], quote=e['quote'], reference_json=e))
    assert len(candidates) == len(expected) == 500
    for actual, exp in zip(candidates, expected):
        actual['reference_json'] = json.loads(actual['reference_json'])
        assert actual == exp
    assert all(c.data_type != 'f' and c.hyperlink is None for ws in book for row in ws for c in row)
    assert db.read_bytes() == before
    return content

def test_real_batch_all_records_reopen_conflict_export(real_batch, tmp_path):
    batch, db, rows, digest = real_batch
    original = batch.read_bytes()
    store = ReviewStore(db, digest)
    for i, r in enumerate(rows):
        if i % 5 == 4:
            continue
        eligible = r['evidence_assessment'].get('implementation_required') is False
        action = 'approved' if eligible and i % 5 == 0 else ('edited','rejected','needs_evidence')[i % 3]
        save(store, r, action, answer=None if action=='approved' else '=HYPERLINK("https://example.invalid","draft")')
    operational = [r for r in rows if r['evidence_assessment'].get('implementation_required') is not False]
    assert len(operational) == 61
    for r in operational:
        prior = store.current(r)
        with pytest.raises(ValueError, match='approval is blocked'):
            save(store, r, 'approved', revision=prior['revision'])
        assert store.current(r) == prior
    r = rows[0]
    stale = store.current(r)['revision']
    other = ReviewStore(db, digest)
    save(other, r, 'edited', revision=stale, answer='+Concurrent draft')
    with pytest.raises(ReviewConflict):
        save(store, r, 'rejected', revision=stale, answer='stale')
    assert store.current(r)['answer'] == '+Concurrent draft'
    save(store, r, 'needs_evidence', revision=stale+1, answer='-Recovered draft')
    reopened = ReviewStore(db, digest)
    assert reopened.current(r)['revision'] == stale+2
    assert len(reopened.history(r['id'])) == stale+2
    content = reconcile(batch, db, rows, digest)
    output = tmp_path/'validated.xlsx'
    assert main([str(batch), str(db), str(output)]) == 0
    assert output.read_bytes() == content
    assert batch.read_bytes() == original
    counts = Counter(reopened.current(r)['status'] for r in rows)
    print('INTEGRATION_COUNTS', dict(sorted(counts.items())), 'operational_blocks=61 candidates=500')

def test_real_batch_streamlit_actions_and_stale_recovery(real_batch, tmp_path, monkeypatch):
    AppTest = pytest.importorskip('streamlit.testing.v1').AppTest
    from test_review_ui_reliability import el, click, edit
    batch, db, rows, digest = real_batch
    monkeypatch.chdir(tmp_path)
    def new():
        app = AppTest.from_file(str(ROOT/'src/trustquery/review_app.py'), default_timeout=60)
        app.session_state['review_batch_path'] = str(batch)
        app.session_state['review_state_path'] = str(db)
        app.run()
        assert not app.exception
        return app
    a, b = new(), new()
    r = rows[0]
    assert r['evidence_assessment']['implementation_required'] is False
    a.multiselect[0].select(r['evidence_references'][0]['chunk_id']).run()
    click(a, 'Build bounded policy extract')
    el(a, 'text_input', 'Reviewer name').set_value('UI reviewer')
    el(a, 'text_area', 'Review rationale / missing evidence / rejection reason').set_value('Bounded synthetic extract reviewed')
    a.checkbox[0].check().run()
    click(a, 'Approve policy extract')
    store = ReviewStore(db, digest)
    assert store.current(r)['status'] == 'approved'
    edit(b, '=Unsaved stale draft', 'UI second')
    click(b, 'Save edit')
    assert any('Review changed' in e.value for e in b.error)
    assert store.current(r)['revision'] == 1
    click(b, 'Keep my draft against compared revision')
    click(b, 'Save edit')
    assert store.current(r)['answer'] == '=Unsaved stale draft'
    el(b, 'text_area', 'Review rationale / missing evidence / rejection reason').set_value('Reject unsupported answer')
    click(b, 'Reject')
    assert store.current(r)['status'] == 'rejected'
    click(b, 'Needs evidence')
    assert store.current(r)['status'] == 'needs_evidence'
    fresh = new()
    assert el(fresh, 'text_area', 'Answer draft').value == '=Unsaved stale draft'
    # Operational record cannot be approved even with an exact extract and checked scope.
    el(fresh, 'selectbox', 'Question').select(rows[1]['id']).run()
    fresh.multiselect[0].select(rows[1]['evidence_references'][0]['chunk_id']).run()
    click(fresh, 'Build bounded policy extract')
    el(fresh, 'text_input', 'Reviewer name').set_value('UI reviewer')
    el(fresh, 'text_area', 'Review rationale / missing evidence / rejection reason').set_value('No operating proof')
    fresh.checkbox[0].check().run()
    click(fresh, 'Approve policy extract')
    assert any('approval is blocked' in e.value for e in fresh.error)
    assert store.current(rows[1])['revision'] == 0
    reconcile(batch, db, rows, digest)

