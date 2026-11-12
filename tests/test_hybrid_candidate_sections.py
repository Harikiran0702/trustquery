import json
import pytest
from trustquery import hybrid_candidate_sections as hybrid
from trustquery import candidate_sections as base, review_policy
from test_candidate_sections import row
from test_review_policy import ROOT, element, local_app


def assert_boundary(result):
    assert result['answer']=='' and result['citations']==[]
    assert result['status']=='needs_evidence'
    assert result['draft_eligible'] is result['human_approval'] is result['operational_proof'] is False


def test_disabled_never_initializes_models(monkeypatch):
    monkeypatch.setattr(hybrid,'models',lambda: pytest.fail('model initialized'))
    assert_boundary(hybrid.locate(row()))
    assert not hybrid.locate(row())['candidates']


@pytest.mark.parametrize('question',[
    'Describe the proposed vendor policy and personnel transfer requirements together.',
    'What does the proposed policy require for review?',
    'Confirm the proposed vendor policy permits bypassing safeguards.',
    'Quote the draft backup policy without incident holds.',
    'Does the proposed encryption policy prove encryption is deployed?',
    'Describe the proposed policy for parking reimbursements.',
])
def test_ambiguous_contradictory_negative_guards(question,monkeypatch):
    monkeypatch.setattr(hybrid,'rank',lambda *a: pytest.fail('unsafe ranking'))
    result=hybrid.locate(row(question),enabled=True)
    assert_boundary(result)
    assert not result['candidates']


def test_collision_and_rank_binding_fail_closed(monkeypatch):
    sections, excluded, hashes=base.load_sections()
    monkeypatch.setattr(base,'load_sections',lambda:(sections+[sections[0]],excluded,hashes))
    result=hybrid.locate(row(),enabled=True)
    assert_boundary(result)
    assert not result['candidates'] and 'collision' in result['reason']


def test_ranked_text_mutation_is_atomic(monkeypatch):
    section=base.load_sections()[0][-1]
    monkeypatch.setattr(hybrid,'rank',lambda *a:[dict(section,text='altered')])
    result=hybrid.locate(row(),enabled=True)
    assert_boundary(result)
    assert not result['candidates'] and 'binding mismatch' in result['reason']


@pytest.mark.parametrize('name',['vendor_risk_management_policy.pdf','access_control_policy.json'])
def test_source_hash_drift_no_model_call(tmp_path,monkeypatch,name):
    for n in json.loads(base.MANIFEST.read_bytes())['files']:
        (tmp_path/n).write_bytes((review_policy.POLICY_DIRECTORY/n).read_bytes())
    monkeypatch.setattr(review_policy,'POLICY_DIRECTORY',tmp_path)
    (tmp_path/name).write_bytes((tmp_path/name).read_bytes()+b' ')
    monkeypatch.setattr(hybrid,'rank',lambda *a:pytest.fail('rank after drift'))
    result=hybrid.locate(row(),enabled=True)
    assert_boundary(result)
    assert not result['candidates'] and 'hash mismatch' in result['reason']


def test_manifest_drift_no_model_call(tmp_path,monkeypatch):
    path=tmp_path/'manifest.json'
    path.write_bytes(base.MANIFEST.read_bytes()+b' ')
    monkeypatch.setattr(base,'MANIFEST',path)
    result=hybrid.locate(row(),enabled=True)
    assert not result['candidates'] and 'manifest changed' in result['reason']


def test_missing_cached_model_abstains(monkeypatch):
    def unavailable(): raise OSError('cached model unavailable')
    monkeypatch.setattr(hybrid,'models',unavailable)
    result=hybrid.locate(row(),enabled=True)
    assert_boundary(result)
    assert not result['candidates'] and 'cached model unavailable' in result['reason']


def test_post_inference_drift_abstains(monkeypatch):
    load=base.load_sections
    count=0
    def check():
        nonlocal count
        count+=1
        if count==3:
            raise ValueError('Source hash mismatch during inference')
        return load()
    monkeypatch.setattr(base,'load_sections',check)
    monkeypatch.setattr(hybrid,'rank',lambda *a:[])
    result=hybrid.locate(row(),enabled=True)
    assert not result['candidates'] and 'during inference' in result['reason']


@pytest.mark.model
def test_live_offline_rank_and_deterministic_evidence():
    question='Describe the proposed personnel policy duties when a worker transfers roles or takes extended leave.'
    result=hybrid.locate(row(question),enabled=True)
    assert_boundary(result)
    assert result['candidates'], result['reason']
    assert result==hybrid.locate(row(question),enabled=True)
    first=result['candidates'][0]
    assert (first['source'],first['page'])==('hr_security_policy.pdf',4)
    for c in result['candidates']:
        assert c['pdf_sha256']==result['corpus_sha256'][c['source']]
        assert c['section_utf8_sha256']==base.sha(c['section_text'].encode())
        assert c['section_text'] in c['document_context'][c['page']-1]['quote']
        assert all(not r['independently_verified'] for r in c['document_context'])
    assert result['locator_status']=='unadjudicated_similarity_candidates'


@pytest.mark.model
def test_hybrid_ui_keeps_review_and_export_separate(tmp_path,monkeypatch):
    from streamlit.testing.v1 import AppTest
    from trustquery.review_store import ReviewStore,load_batch
    from trustquery.review_export import export_bytes
    from test_review_export import read
    monkeypatch.chdir(tmp_path)
    batch=tmp_path/'data/batch_runs/major_deliverable_1/results.json'
    batch.parent.mkdir(parents=True)
    r=row()
    batch.write_text(json.dumps([r]))
    db=tmp_path/'data/reviews/major_deliverable_2.sqlite3'
    store=ReviewStore(db,load_batch(batch)[1])
    app=local_app(AppTest, str(ROOT/'src/trustquery/review_app.py'),default_timeout=90).run()
    element(app,'text_area','Answer draft').set_value('Unsaved human draft').run()
    element(app,'toggle','Enable read-only section candidates').set_value(True).run()
    element(app,'toggle','Use offline hybrid section ranking').set_value(True).run()
    element(app,'button','Locate full policy sections (read only)').click().run()
    assert not app.exception
    assert any('UNVERIFIED SECTION' in e.label for e in app.expander)
    assert element(app,'text_area','Answer draft').value=='Unsaved human draft'
    assert store.current(r)['revision']==0
    element(app,'text_input','Reviewer name').set_value('Human test reviewer').run()
    element(app,'button','Save edit').click().run()
    saved=store.current(r)
    element(app,'button','Locate full policy sections (read only)').click().run()
    assert store.current(r)==saved and saved['citations']==[]
    _,verified,candidates,_=read(export_bytes(batch,db))
    assert len(verified)==1 and len(candidates)==1
