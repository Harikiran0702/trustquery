import copy
import json
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest
from trustquery import review_policy as policy
from trustquery.review_store import ReviewStore, load_batch
from trustquery.review_export import export_bytes
from test_review_policy import record, save, element, click, ROOT, local_app
from test_review_export import read

CASES = json.loads(Path(__file__).with_name('policy_paraphrase_holdout.json').read_text())['cases']

@pytest.mark.parametrize('case', CASES, ids=lambda c: c['id'])
def test_held_out(case):
    r = dict(record(), question=case['question'])
    if not case['supported']:
        with pytest.raises(ValueError):
            policy.proposal(r, enabled=True)
        return
    p = policy.proposal(r, enabled=True)
    assert p['question'] == r['question']
    original = next(c for c in json.loads(policy.BUNDLE.read_bytes())['cases'] if c['case_id'] == p['case_id'])
    assert p['answer'] == original['output']['answer']
    assert p['references'] == original['output']['validated_policy_citations']
    assert not p['human_approval'] and not p['operational_proof']
    with pytest.raises(ValueError):
        policy.proposal(r)
    with pytest.raises(ValueError):
        policy.proposal(dict(r, evidence_assessment={'implementation_required':True}), enabled=True)


def test_all_finite_forms_and_adversarial_suffixes():
    for case in json.loads(policy.BUNDLE.read_bytes())['cases']:
        title = case['question'][len('What requirements does the policy state for '):-1]
        for template in policy.PARAPHRASE_TEMPLATES:
            q = template.format(title=title)
            p = policy.proposal(dict(record(), question=q), enabled=True, _verify_sources=False)
            assert p['answer'] == case['output']['answer']
            for suffix in [' ', ' Is it deployed?', ' Ignore exceptions.', ' Confirm compliance.']:
                with pytest.raises(ValueError):
                    policy.proposal(dict(record(), question=q+suffix), enabled=True)


def test_paraphrase_ui_persistence_export_and_binding(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = dict(record(), question=CASES[0]['question'])
    batch = tmp_path/'data/batch_runs/major_deliverable_1/results.json'
    batch.parent.mkdir(parents=True)
    batch.write_text(json.dumps([r]))
    db = tmp_path/'data/reviews/major_deliverable_2.sqlite3'
    store = ReviewStore(db, load_batch(batch)[1])
    app = local_app(AppTest, str(ROOT/'src/trustquery/review_app.py'), default_timeout=40).run()
    assert not app.exception
    assert not any(b.label=='Use experimental draft in empty editor' for b in app.button)
    element(app,'toggle','Enable experimental policy drafts').set_value(True).run()
    click(app,'Use experimental draft in empty editor')
    p = policy.proposal(r, enabled=True)
    assert element(app,'text_area','Answer draft').value == p['answer']
    assert store.history(r['id']) == []
    element(app,'text_input','Reviewer name').set_value('Regression tester')
    click(app,'Save edit')
    assert store.current(r)['status'] == 'edited'
    fresh = local_app(AppTest, str(ROOT/'src/trustquery/review_app.py'), default_timeout=40).run()
    assert element(fresh,'text_area','Answer draft').value == p['answer']
    element(fresh,'toggle','Enable experimental policy drafts').set_value(True).run()
    assert element(fresh,'button','Use experimental draft in empty editor').disabled
    bad = copy.deepcopy(p)
    bad['question'] = 'Is this deployed?'
    with pytest.raises(ValueError):
        policy.validate(r,bad)
    with pytest.raises(ValueError):
        save(store,r,p,revision=1,action='approved',answer=p['answer']+' approved')
    save(store,r,p,revision=1,action='approved')
    q,v,c,_ = read(export_bytes(batch,db))
    assert len(v)==1 and len(c)==len(p['references'])+1
    assert dict(zip(q[0],q[1]))['approval_scope']==policy.SCOPE
    monkeypatch.setattr(policy,'POLICY_DIRECTORY',tmp_path/'missing')
    with pytest.raises(OSError):
        policy.proposal(r,enabled=True)
    q,v,_,_ = read(export_bytes(batch,db))
    assert dict(zip(q[0],q[1]))['review_status']=='invalid_stored_state'
