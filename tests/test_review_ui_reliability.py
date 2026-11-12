"""Streamlit sessions share only SQLite; all review databases are temporary."""
import json
from pathlib import Path
import pytest
from trustquery.review_store import ReviewStore, load_batch


def el(app, kind, label):
    return next(e for e in getattr(app, kind) if e.label == label)


def click(app, label):
    el(app, 'button', label).click().run()
    assert not app.exception


def edit(app, text, reviewer='Alice'):
    el(app, 'text_area', 'Answer draft').set_value(text)
    el(app, 'text_input', 'Reviewer name').set_value(reviewer)


@pytest.fixture
def ui(tmp_path, monkeypatch):
    AppTest = pytest.importorskip('streamlit.testing.v1').AppTest
    monkeypatch.chdir(tmp_path)
    batch = tmp_path/'data/batch_runs/major_deliverable_1/results.json'
    batch.parent.mkdir(parents=True)
    rows = [dict(id=f'q{i}', question=f'Question {i}', answer='', outcome='abstained',
                 metadata={'domain': f'd{i}'}, evidence_assessment={'implementation_required': False},
                 evidence_references=[dict(source='p.pdf', page=1, chunk_index=0, chunk_id='c1', quote='Policy requires MFA.')]) for i in (1, 2)]
    batch.write_text(json.dumps(rows))
    app = Path(__file__).resolve().parents[1]/'src/trustquery/review_app.py'
    def new():
        at = local_app(AppTest, str(app), default_timeout=30).run()
        assert not at.exception
        return at
    _, digest = load_batch(batch)
    store = ReviewStore(tmp_path/'data/reviews/major_deliverable_2.sqlite3', digest)
    return new, store, rows


def test_two_sessions_stale_save_reruns_and_repeated_conflict(ui):
    new, store, rows = ui
    a, b = new(), new()
    edit(a, 'Alice draft')
    a.run()
    edit(b, 'Bob first', 'Bob')
    click(b, 'Save edit')
    a.run()  # An unrelated rerun must not adopt Bob's revision or widgets.
    assert el(a, 'text_area', 'Answer draft').value == 'Alice draft'
    assert any('Revision: 0' in c.value for c in a.caption)
    click(a, 'Save edit')
    assert store.current(rows[0])['answer'] == 'Bob first'
    assert len(store.history('q1')) == 1
    assert el(a, 'text_area', 'Answer draft').value == 'Alice draft'
    comparison = next(j.value for j in a.json if 'Originally displayed' in j.value)
    assert 'Bob first' in comparison and 'Alice draft' in comparison
    assert el(a, 'button', 'Save edit').disabled
    a.run()
    assert el(a, 'text_area', 'Answer draft').value == 'Alice draft'
    # Bob saves again after Alice saw the comparison, before she resolves it.
    edit(b, 'Bob second', 'Bob')
    click(b, 'Save edit')
    click(a, 'Keep my draft against compared revision')
    assert any('Revision: 1' in c.value for c in a.caption)
    assert not a.checkbox[0].value
    click(a, 'Save edit')
    assert len(store.history('q1')) == 2
    assert any('Review changed' in e.value for e in a.error)
    click(a, 'Keep my draft against compared revision')
    click(a, 'Save edit')
    assert store.current(rows[0])['revision'] == 3
    assert store.current(rows[0])['answer'] == 'Alice draft'
    assert store.current(rows[0])['reviewer'] == 'Alice'


def test_conflict_discard_loads_compared_state(ui):
    new, store, rows = ui
    a, b = new(), new()
    edit(a, 'Unsaved Alice')
    edit(b, 'Stored Bob', 'Bob')
    click(b, 'Save edit')
    click(a, 'Save edit')
    click(a, 'Discard my draft and load compared revision')
    assert el(a, 'text_area', 'Answer draft').value == 'Stored Bob'
    assert el(a, 'text_input', 'Reviewer name').value == 'Bob'
    assert len(store.history('q1')) == 1
    assert not el(a, 'button', 'Save edit').disabled


@pytest.mark.parametrize('navigation', ['Next', 'Question', 'Search ID or question', 'Review status', 'Domain', 'Pipeline status'])
def test_dirty_navigation_cancel_and_discard(ui, navigation):
    new, store, rows = ui
    a = new()
    edit(a, 'Retain every field')
    el(a, 'multiselect', 'Passages to review (selection does not verify citations)').select('c1').run()
    el(a, 'text_area', 'Review rationale / missing evidence / rejection reason').set_value('My rationale')
    a.checkbox[0].check()
    if navigation == 'Next':
        click(a, 'Next')
    elif navigation == 'Question':
        el(a, 'selectbox', navigation).select('q2').run()
    elif navigation == 'Search ID or question':
        el(a, 'text_input', navigation).set_value('no match').run()
    else:
        option = {'Review status': 'approved', 'Domain': 'd2', 'Pipeline status': 'Abstained'}[navigation]
        el(a, 'selectbox', navigation).select(option).run()
    assert not a.exception
    assert el(a, 'text_area', 'Answer draft').value == 'Retain every field'
    a.run()
    click(a, 'Cancel navigation')
    assert el(a, 'selectbox', 'Question').value == 'q1'
    assert el(a, 'text_input', 'Search ID or question').value == ''
    assert el(a, 'selectbox', 'Review status').value == 'All'
    assert el(a, 'text_area', 'Review rationale / missing evidence / rejection reason').value == 'My rationale'
    assert a.checkbox[0].value and a.multiselect[0].value == ['c1']
    click(a, 'Next')
    click(a, 'Discard changes and continue')
    assert el(a, 'selectbox', 'Question').value == 'q2'
    click(a, 'Previous')
    assert el(a, 'text_area', 'Answer draft').value == ''
    assert store.history('q1') == []


def test_save_navigation_validation_and_conflict(ui):
    new, store, rows = ui
    a, b = new(), new()
    edit(a, 'Alice draft', '')
    click(a, 'Next')
    click(a, 'Save edit and continue')
    assert any('Reviewer name' in e.value for e in a.error)
    assert store.history('q1') == []
    el(a, 'text_input', 'Reviewer name').set_value('Alice').run()
    edit(b, 'Bob draft', 'Bob')
    click(b, 'Save edit')
    click(a, 'Save edit and continue')
    assert el(a, 'text_area', 'Answer draft').value == 'Alice draft'
    assert len(store.history('q1')) == 1
    click(a, 'Keep my draft against compared revision')
    click(a, 'Save edit and continue')
    assert el(a, 'selectbox', 'Question').value == 'q2'
    click(a, 'Previous')
    assert el(a, 'text_area', 'Answer draft').value == 'Alice draft'
    assert store.current(rows[0])['revision'] == 2
    fresh = new()
    assert el(fresh, 'text_area', 'Answer draft').value == 'Alice draft'


def test_clean_session_still_saves_original_revision(ui):
    new, store, rows = ui
    a, b = new(), new()
    edit(b, 'Bob', 'Bob')
    click(b, 'Save edit')
    a.run()
    edit(a, 'Alice')
    click(a, 'Save edit')
    assert len(store.history('q1')) == 1
    assert any('Review changed' in e.value for e in a.error)


@pytest.mark.parametrize('action', ['Approve policy extract', 'Reject', 'Needs evidence'])
def test_stale_decisions_keep_all_draft_fields(ui, action):
    new, store, rows = ui
    a, b = new(), new()
    a.multiselect[0].select('c1').run()
    click(a, 'Build bounded policy extract')
    original_answer = el(a, 'text_area', 'Answer draft').value
    el(a, 'text_input', 'Reviewer name').set_value('Alice')
    el(a, 'text_area', 'Review rationale / missing evidence / rejection reason').set_value('Full scope reviewed')
    a.checkbox[0].check().run()
    edit(b, 'Bob', 'Bob')
    click(b, 'Save edit')
    click(a, action)
    assert len(store.history('q1')) == 1
    assert el(a, 'text_area', 'Answer draft').value == original_answer
    assert a.multiselect[0].value == ['c1'] and a.checkbox[0].value
    assert el(a, 'text_area', 'Review rationale / missing evidence / rejection reason').value == 'Full scope reviewed'
    click(a, 'Keep my draft against compared revision')
    assert not a.checkbox[0].value
    assert el(a, 'text_area', 'Answer draft').value == original_answer


def test_question_draft_revision_survives_widget_cleanup(ui):
    new, store, rows = ui
    a, b = new(), new()
    click(a, 'Next')
    edit(b, 'Bob', 'Bob')
    click(b, 'Save edit')
    a.run()
    click(a, 'Previous')
    assert el(a, 'text_area', 'Answer draft').value == ''
    assert any('Revision: 0' in c.value for c in a.caption)
    edit(a, 'Alice')
    click(a, 'Save edit')
    assert len(store.history('q1')) == 1
    assert any('Review changed' in e.value for e in a.error)


@pytest.mark.parametrize('resolution', ['Save edit and continue', 'Discard changes and continue'])
def test_empty_filter_guard_completion(ui, resolution):
    new, store, rows = ui
    a = new()
    edit(a, 'Draft')
    el(a, 'text_input', 'Search ID or question').set_value('missing').run()
    assert el(a, 'text_area', 'Answer draft').value == 'Draft'
    click(a, resolution)
    assert any('No questions match' in i.value for i in a.info)
    el(a, 'text_input', 'Search ID or question').set_value('').run()
    assert not a.exception
    expected = 'Draft' if resolution.startswith('Save') else ''
    assert el(a, 'text_area', 'Answer draft').value == expected
    assert len(store.history('q1')) == int(resolution.startswith('Save'))


def test_unsaved_edits_block_same_request_context_switch(ui, tmp_path):
    new, store, rows = ui
    a = new()
    original_path = el(a, 'text_input', 'Review database').value
    edit(a, 'Retained across attempted path change')
    el(a, 'text_input', 'Review database').set_value(str(tmp_path/'other.sqlite3')).run()
    assert not a.exception
    assert el(a, 'text_input', 'Review database').value == original_path
    assert el(a, 'text_input', 'Review database').disabled
    assert el(a, 'text_area', 'Answer draft').value == 'Retained across attempted path change'
    assert not (tmp_path/'other.sqlite3').exists()


def test_save_normalization_is_not_unsaved_work(ui):
    new, store, rows = ui
    a = new()
    edit(a, 'Draft', ' Alice ')
    el(a, 'text_area', 'Review rationale / missing evidence / rejection reason').set_value(' reason ')
    click(a, 'Save edit')
    assert el(a, 'text_input', 'Reviewer name').value == 'Alice'
    assert el(a, 'text_area', 'Review rationale / missing evidence / rejection reason').value == 'reason'
    click(a, 'Next')
    assert el(a, 'selectbox', 'Question').value == 'q2'
    assert not any('Unsaved edits:' in w.value for w in a.warning)


def local_app(AppTest, *args, **kwargs):
    app = AppTest.from_file(*args, **kwargs)
    app.session_state['review_batch_path'] = 'data/batch_runs/major_deliverable_1/results.json'
    app.session_state['review_state_path'] = 'data/reviews/major_deliverable_2.sqlite3'
    return app
