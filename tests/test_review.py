import copy
import json
from pathlib import Path
import pytest
from trustquery.review_store import ReviewStore, load_batch, policy_extract


def record():
    return dict(id='q1', question='Does policy require MFA?', answer='', outcome='abstained', evidence_assessment={'implementation_required':False}, evidence_references=[dict(source='policy.pdf',page=1,chunk_index=0,chunk_id='policy.pdf:p1:c0',quote='Policy must require MFA.')])


def save(store, r, **changes):
    args=dict(revision=0,action='edited',answer='Draft',selected=[],reviewer='Reviewer',note='',scope_checked=False)
    args.update(changes)
    return store.save(r, **args)


def test_restart_history_and_edit_revokes_approval(tmp_path):
    r=record()
    path=tmp_path/'review.sqlite3'
    store=ReviewStore(path,'a')
    original=copy.deepcopy(r)
    answer=policy_extract(r,['policy.pdf:p1:c0'])
    approved=save(store,r,action='approved',answer=answer,selected=['policy.pdf:p1:c0'],note='Reviewed full question and scope',scope_checked=True)
    assert approved['citations']==[] and not approved['operational_proof']
    restarted=ReviewStore(path,'a')
    assert restarted.current(r)==approved
    edited=save(restarted,r,revision=1)
    assert edited['status']=='edited' and edited['approval_scope'] is None
    assert len(restarted.history('q1'))==2
    assert r==original


@pytest.mark.parametrize('change', [dict(answer='Yes, MFA is deployed.'),dict(scope_checked=False),dict(note=''),dict(selected=[]),dict(reviewer='')])
def test_approval_guard(tmp_path,change):
    r=record()
    store=ReviewStore(tmp_path/'r.sqlite3','a')
    args=dict(action='approved',answer=policy_extract(r,['policy.pdf:p1:c0']),selected=['policy.pdf:p1:c0'],note='Scope reviewed',scope_checked=True)
    args.update(change)
    with pytest.raises(ValueError):
        save(store,r,**args)
    assert store.current(r)['revision']==0


@pytest.mark.parametrize('assessment', [{}, {'implementation_required':True}])
def test_operational_and_unknown_blocked(tmp_path,assessment):
    r=record()
    r['evidence_assessment']=assessment
    store=ReviewStore(tmp_path/'r.sqlite3','a')
    with pytest.raises(ValueError,match='Operational'):
        save(store,r,action='approved',answer=policy_extract(r,['policy.pdf:p1:c0']),selected=['policy.pdf:p1:c0'],note='Checked',scope_checked=True)


def test_batch_binding_and_conflict(tmp_path):
    r=record()
    path=tmp_path/'r.sqlite3'
    store=ReviewStore(path,'a')
    other=ReviewStore(path,'a')
    save(store,r)
    with pytest.raises(ValueError,match='another session'):
        save(other,r)
    with pytest.raises(ValueError,match='different batch'):
        ReviewStore(path,'b')
    assert len(store.history('q1'))==1


@pytest.mark.parametrize('action',['rejected','needs_evidence'])
def test_reason_required_and_persisted(tmp_path,action):
    r=record()
    store=ReviewStore(tmp_path/'r.sqlite3','a')
    with pytest.raises(ValueError):
        save(store,r,action=action)
    saved=save(store,r,action=action,note='Need an access review record')
    assert store.current(r)==saved


def test_invalid_and_duplicate_input(tmp_path):
    path=tmp_path/'batch.json'
    path.write_text(json.dumps([record(),record()]))
    with pytest.raises(ValueError):
        load_batch(path)
    r=record()
    r['evidence_references'][0]['page']=0
    path.write_text(json.dumps([r]))
    with pytest.raises(ValueError):
        load_batch(path)
    with pytest.raises(ValueError):
        ReviewStore(tmp_path/'batch.json','a')


def test_real_batch_readonly_and_cdq021(tmp_path):
    path=Path(__file__).resolve().parents[1]/'data/batch_runs/major_deliverable_1/results.json'
    before=path.read_bytes()
    rows,digest=load_batch(path)
    assert len(rows)==100 and sum(len(r['evidence_references']) for r in rows)==500
    r=next(r for r in rows if r['id']=='cdq021')
    store=ReviewStore(tmp_path/'r.sqlite3',digest)
    save(store,r,action='needs_evidence',note='Pending human evidence review')
    assert path.read_bytes()==before and store.current(r)['citations']==[]


def test_streamlit_navigation_filters_edit_restart(tmp_path,monkeypatch):
    AppTest = pytest.importorskip('streamlit.testing.v1').AppTest
    monkeypatch.chdir(tmp_path)
    batch=tmp_path/'data/batch_runs/major_deliverable_1'
    batch.mkdir(parents=True)
    rows=[record(),dict(record(),id='q2',question='Other question')]
    (batch/'results.json').write_text(json.dumps(rows))
    app=Path(__file__).resolve().parents[1]/'src/trustquery/review_app.py'
    at=local_app(AppTest, str(app), default_timeout=30).run()
    assert not at.exception
    def by_label(elements,label): return next(e for e in elements if e.label==label)
    by_label(at.text_area,'Answer draft').set_value('My draft')
    by_label(at.text_input,'Reviewer name').set_value('Reviewer')
    by_label(at.button,'Save edit').click().run()
    assert not at.exception
    assert by_label(at.text_area,'Answer draft').value=='My draft'
    by_label(at.button,'Next').click().run()
    assert by_label(at.selectbox,'Question').value=='q2'
    by_label(at.button,'Previous').click().run()
    assert by_label(at.text_area,'Answer draft').value=='My draft'
    at=local_app(AppTest, str(app), default_timeout=30).run()
    assert not at.exception and by_label(at.text_area,'Answer draft').value=='My draft'
    by_label(at.selectbox,'Review status').select('approved').run()
    assert any('No questions' in x.value for x in at.info)


def test_streamlit_approval_and_rejection(tmp_path,monkeypatch):
    AppTest = pytest.importorskip('streamlit.testing.v1').AppTest
    monkeypatch.chdir(tmp_path)
    batch=tmp_path/'data/batch_runs/major_deliverable_1'
    batch.mkdir(parents=True)
    (batch/'results.json').write_text(json.dumps([record()]))
    app=Path(__file__).resolve().parents[1]/'src/trustquery/review_app.py'
    at=local_app(AppTest, str(app), default_timeout=30).run()
    def el(kind,label): return next(e for e in getattr(at,kind) if e.label==label)
    el('multiselect','Passages to review (selection does not verify citations)').select('policy.pdf:p1:c0').run()
    el('button','Build bounded policy extract').click().run()
    el('text_input','Reviewer name').set_value('Reviewer')
    el('text_area','Review rationale / missing evidence / rejection reason').set_value('Checked all scope and qualifiers')
    at.checkbox[0].check()
    el('button','Approve policy extract').click().run()
    assert not at.exception and any('Review: approved' in x.value for x in at.caption)
    assert {m.label: m.value for m in at.metric}['Approved'] == '1'
    el('button','Reject').click().run()
    assert not at.exception and any('Review: rejected' in x.value for x in at.caption)
    metrics = {m.label: m.value for m in at.metric}
    assert metrics['Approved'] == '0' and metrics['Rejected'] == '1'


def test_streamlit_real_batch_all_candidates(tmp_path):
    AppTest = pytest.importorskip('streamlit.testing.v1').AppTest
    root=Path(__file__).resolve().parents[1]
    at=AppTest.from_file(str(root/'src/trustquery/review_app.py'),default_timeout=30)
    at.session_state['review_batch_path'] = str(root/'data/batch_runs/major_deliverable_1/results.json')
    at.session_state['review_state_path'] = str(tmp_path/'review.sqlite3')
    at.run()
    assert not at.exception
    selector=next(e for e in at.selectbox if e.label=='Question')
    assert len(selector.options)==100
    selector.select('cdq021').run()
    assert not at.exception
    rows,_=load_batch(root/'data/batch_runs/major_deliverable_1/results.json')
    row=next(r for r in rows if r['id']=='cdq021')
    assert at.multiselect[0].options == ['None \u2014 none of these passages support the answer'] + [
        f"{ref['source']} — page {ref['page']} (passage {index})"
        for index, ref in enumerate(row['evidence_references'], 1)
    ]
    next(e for e in at.text_input if e.label=='Reviewer name').set_value('Reviewer')
    next(e for e in at.text_area if e.label=='Review rationale / missing evidence / rejection reason').set_value('Need operational corroboration')
    next(e for e in at.button if e.label=='Needs evidence').click().run()
    assert not at.exception and any('Review: needs_evidence' in x.value for x in at.caption)


@pytest.mark.parametrize('outcome',['processing_error','implementation_evidence_required'])
def test_inconsistent_outcome_cannot_approve(tmp_path,outcome):
    r=record()
    r['outcome']=outcome
    store=ReviewStore(tmp_path/'r.sqlite3','a')
    with pytest.raises(ValueError):
        save(store,r,action='approved',answer=policy_extract(r,['policy.pdf:p1:c0']),selected=['policy.pdf:p1:c0'],note='Checked',scope_checked=True)


def local_app(AppTest, *args, **kwargs):
    app = AppTest.from_file(*args, **kwargs)
    app.session_state['review_batch_path'] = 'data/batch_runs/major_deliverable_1/results.json'
    app.session_state['review_state_path'] = 'data/reviews/major_deliverable_2.sqlite3'
    return app
