import copy
import json
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest
from trustquery import review_policy as policy
from trustquery.review_store import ReviewStore, ReviewConflict, load_batch
from trustquery.review_export import export_bytes
from test_review_export import read

ROOT = Path(__file__).resolve().parents[1]

def record():
    case = json.loads(policy.BUNDLE.read_bytes())['cases'][0]
    return dict(id='local-section', question=case['question'], answer='', outcome='abstained',
                metadata={}, evidence_assessment={'implementation_required':False}, evidence_references=[])

def element(app, kind, label):
    return next(e for e in getattr(app, kind) if e.label == label)

def click(app, label):
    element(app,'button',label).click().run()
    assert not app.exception

@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    batch=tmp_path/'data/batch_runs/major_deliverable_1/results.json'
    batch.parent.mkdir(parents=True)
    r=record()
    batch.write_text(json.dumps([r]))
    _, digest=load_batch(batch)
    db=tmp_path/'data/reviews/major_deliverable_2.sqlite3'
    return r,batch,db,ReviewStore(db,digest)

def save(store,r,p,**kwargs):
    args=dict(revision=0,action='edited',answer=p['answer'],experimental_policy=p,
              selected=[],reviewer='Human tester',note='Inspected full scope',scope_checked=True)
    args.update(kwargs)
    return store.save(r,**args)

def test_binding_and_abstention(tmp_path,monkeypatch):
    r=record()
    with pytest.raises(ValueError):
        policy.proposal(r)
    for change in [dict(question=r['question']+' '),dict(question='Is MFA deployed?'),
                   dict(evidence_assessment={'implementation_required':True}),dict(outcome='processing_error')]:
        with pytest.raises(ValueError):
            policy.proposal(dict(r,**change),enabled=True)
    p=policy.proposal(r,enabled=True)
    assert p['human_approval'] is False and p['operational_proof'] is False
    assert len(p['references']) == 6
    bad=copy.deepcopy(p)
    bad['references'][0]['page']=99
    with pytest.raises(ValueError):
        policy.validate(r,bad)
    monkeypatch.setattr(policy,'POLICY_DIRECTORY',tmp_path)
    for source in p['corpus_sha256']:
        (tmp_path/source).write_bytes(b'%PDF-tampered')
    with pytest.raises(ValueError):
        policy.proposal(r,enabled=True)

def test_store_export_and_revision(setup):
    r,batch,db,store=setup
    p=policy.proposal(r,enabled=True)
    with pytest.raises(ValueError):
        save(store,r,p,action='approved',answer='AI says deployed')
    with pytest.raises(ValueError):
        save(store,r,p,action='approved',scope_checked=False)
    state=save(store,r,p)
    assert state['status']=='edited' and state['citations']==[]
    with pytest.raises(ReviewConflict):
        save(store,r,p,action='approved')
    state=save(store,r,p,revision=1,action='approved')
    assert state['approval_scope']==policy.SCOPE
    q,v,c,_=read(export_bytes(batch,db))
    row=dict(zip(q[0],q[1]))
    assert row['review_status']=='approved'
    assert row['approval_scope']==policy.SCOPE and len(v)==1
    assert len(c)==7 and all('UNVERIFIED' in x[1] for x in c[1:])
    assert ReviewStore(db,load_batch(batch)[1]).current(r)==state
    save(store,r,p,revision=2,answer=p['answer']+' edited')
    assert store.current(r)['status']=='edited'
    assert dict(zip(q[0],read(export_bytes(batch,db))[0][1]))['approved_policy_extract']==''

@pytest.mark.parametrize('action',['Reject','Needs evidence','Approve policy extract'])
def test_ui_optin_persistence_and_decisions(setup,action):
    r,batch,db,store=setup
    app=local_app(AppTest, str(ROOT/'src/trustquery/review_app.py'),default_timeout=40).run()
    assert not app.exception
    assert not any(b.label=='Use experimental draft in empty editor' for b in app.button)
    element(app,'toggle','Enable experimental policy drafts').set_value(True).run()
    assert not app.exception
    click(app,'Use experimental draft in empty editor')
    expected=policy.proposal(r,enabled=True)['answer']
    assert element(app,'text_area','Answer draft').value==expected
    assert store.history(r['id'])==[]
    app.run()
    assert element(app,'button','Use experimental draft in empty editor').disabled
    element(app,'text_input','Reviewer name').set_value('Alice')
    element(app,'text_area','Review rationale / missing evidence / rejection reason').set_value('Reviewed all context')
    app.checkbox[0].check()
    click(app,action)
    state=store.current(r)
    assert state['status']=={'Reject':'rejected','Needs evidence':'needs_evidence','Approve policy extract':'approved'}[action]
    fresh=local_app(AppTest, str(ROOT/'src/trustquery/review_app.py'),default_timeout=40).run()
    assert not fresh.exception and element(fresh,'text_area','Answer draft').value==expected
    assert any('experimental' in c.value.lower() for c in fresh.caption)
    element(fresh,'toggle','Enable experimental policy drafts').set_value(True).run()
    assert element(fresh,'button','Use experimental draft in empty editor').disabled

def test_ui_no_overwrite_unsaved_and_concurrent(setup):
    r,batch,db,store=setup
    app=local_app(AppTest, str(ROOT/'src/trustquery/review_app.py'),default_timeout=40).run()
    element(app,'text_area','Answer draft').set_value('my existing work').run()
    element(app,'toggle','Enable experimental policy drafts').set_value(True).run()
    assert element(app,'button','Use experimental draft in empty editor').disabled
    assert element(app,'text_area','Answer draft').value=='my existing work'


def test_export_fails_closed_when_source_changes(setup,monkeypatch,tmp_path):
    r,batch,db,store=setup
    p=policy.proposal(r,enabled=True)
    save(store,r,p,action='approved')
    monkeypatch.setattr(policy,'POLICY_DIRECTORY',tmp_path/'missing')
    q,v,_,_=read(export_bytes(batch,db))
    result=dict(zip(q[0],q[1]))
    assert result['review_status']=='invalid_stored_state' and result['approved_policy_extract']=='' and len(v)==1


def test_all_twelve_exact_reuse_and_export(tmp_path):
    cases=json.loads(policy.BUNDLE.read_bytes())['cases']
    rows=[dict(record(),id=c['case_id'],question=c['question']) for c in cases]
    batch=tmp_path/'all.json'
    batch.write_text(json.dumps(rows))
    db=tmp_path/'all.sqlite3'
    store=ReviewStore(db,load_batch(batch)[1])
    for r,c in zip(rows,cases):
        p=policy.proposal(r,enabled=True)
        assert p['answer']==c['output']['answer']
        save(store,r,p,action='approved')
    q,v,c,_=read(export_bytes(batch,db))
    assert len(q)==13 and len(v)==1 and len(c)==69
    assert all(row[2]=='approved' for row in q[1:])


def test_bundle_tamper(tmp_path,monkeypatch):
    path=tmp_path/'bundle.json'
    path.write_bytes(policy.BUNDLE.read_bytes()+b' ')
    monkeypatch.setattr(policy,'BUNDLE',path)
    with pytest.raises(ValueError,match='bundle version'):
        policy.proposal(record(),enabled=True)


def test_ui_experimental_conflict(setup):
    r,batch,db,store=setup
    app=local_app(AppTest, str(ROOT/'src/trustquery/review_app.py'),default_timeout=40).run()
    element(app,'toggle','Enable experimental policy drafts').set_value(True).run()
    click(app,'Use experimental draft in empty editor')
    p=policy.proposal(r,enabled=True)
    save(store,r,p,answer='Another session saved this')
    element(app,'text_input','Reviewer name').set_value('Alice')
    click(app,'Save edit')
    assert any('Review changed' in e.value for e in app.error)
    assert element(app,'text_area','Answer draft').value==p['answer']
    assert store.current(r)['answer']=='Another session saved this'
    click(app,'Keep my draft against compared revision')
    assert not app.checkbox[0].value
    click(app,'Save edit')
    assert store.current(r)['revision']==2
    assert store.current(r)['experimental_policy']==p


def test_drift_allows_feedback_but_blocks_approval(setup,monkeypatch,tmp_path):
    r,batch,db,store=setup
    p=policy.proposal(r,enabled=True)
    save(store,r,p)
    monkeypatch.setattr(policy,'POLICY_DIRECTORY',tmp_path/'missing')
    save(store,r,p,revision=1,action='needs_evidence',note='Source version changed')
    assert store.current(r)['status']=='needs_evidence'
    with pytest.raises(OSError):
        save(store,r,p,revision=2,action='approved')


def test_null_answer_cannot_bypass_approval(setup):
    r,batch,db,store=setup
    p=policy.proposal(r,enabled=True)
    with pytest.raises(ValueError):
        save(store,r,p,action='approved',answer=None)
    assert store.history(r['id'])==[]


def local_app(AppTest, *args, **kwargs):
    app = AppTest.from_file(*args, **kwargs)
    app.session_state['review_batch_path'] = 'data/batch_runs/major_deliverable_1/results.json'
    app.session_state['review_state_path'] = 'data/reviews/major_deliverable_2.sqlite3'
    return app
