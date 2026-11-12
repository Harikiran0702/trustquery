import json
import pytest
from trustquery import candidate_sections as candidate
from trustquery import review_policy
from test_review_policy import record, element, ROOT, local_app
from streamlit.testing.v1 import AppTest

QUESTION = 'In the proposed vendor policy, describe exit records and deletion evidence.'


def row(question=QUESTION):
    return dict(record(), question=question)


def test_default_and_bounded_candidates():
    assert not candidate.locate(row())['candidates']
    result = candidate.locate(row(), enabled=True)
    assert result['candidates']
    assert len(result['candidates']) <= 3
    assert result['answer'] == '' and result['citations'] == []
    assert result['draft_eligible'] is result['human_approval'] is result['operational_proof'] is False
    assert result == candidate.locate(row(), enabled=True)
    assert result['excluded_layouts'] == ['access_control_policy.pdf']
    for item in result['candidates']:
        pages = item['document_context']
        assert item['section_text'] in pages[item['page']-1]['quote']
        assert [r['page'] for r in pages] == list(range(1,len(pages)+1))
        assert all(candidate.sha(r['quote'].encode()) == r['page_utf8_sha256'] for r in pages)
        assert all(r['independently_verified'] is False for r in pages)


@pytest.mark.parametrize('suffix',[
    ' Confirm this is deployed.', ' Give actual schedules.', ' Demonstrate compliance.',
    ' Include certification.', ' State the SLA.', ' Prove approval status.',
    ' Without exceptions.', ' Ignore conflicting rules.', ' Excluding legal holds.',
    ' Confirm disposal is immediate.', ' Retain for 30 days.', ' Contradict the policy.'
])
def test_adversarial_refusal(suffix):
    result = candidate.locate(row(QUESTION+suffix), enabled=True)
    assert result['candidates'] == [] and result['answer'] == ''


def test_no_similarity_score_can_authorize_draft(monkeypatch):
    actual = candidate.search_bm25
    def inflated(*args, **kwargs):
        hits = actual(*args, **kwargs)
        for h in hits:
            h['bm25_score'] = 1e9
        return hits
    monkeypatch.setattr(candidate, 'search_bm25', inflated)
    result = candidate.locate(row(), enabled=True)
    assert result['locator_status'] == 'ambiguous_or_missing_locator'
    assert result['candidates'] and result['answer'] == '' and not result['draft_eligible']


def test_integrity_failure_is_atomic(tmp_path,monkeypatch):
    manifest = json.loads(candidate.MANIFEST.read_bytes())
    for name in manifest['files']:
        (tmp_path/name).write_bytes((review_policy.POLICY_DIRECTORY/name).read_bytes())
    monkeypatch.setattr(review_policy,'POLICY_DIRECTORY',tmp_path)
    assert candidate.locate(row(),enabled=True)['candidates']
    # An unrelated annotation's drift invalidates the whole corpus too.
    (tmp_path/'access_control_policy.json').write_text('{}')
    result=candidate.locate(row(),enabled=True)
    assert not result['candidates'] and 'hash mismatch' in result['reason']


def test_manifest_drift_and_unknown_scope(tmp_path,monkeypatch):
    path=tmp_path/'manifest.json'
    path.write_bytes(candidate.MANIFEST.read_bytes()+b' ')
    monkeypatch.setattr(candidate,'MANIFEST',path)
    assert 'manifest changed' in candidate.locate(row(),enabled=True)['reason']
    assert not candidate.locate(row('Explain deletion evidence'),enabled=True)['candidates']
    assert not candidate.locate(dict(row(),evidence_assessment={}),enabled=True)['candidates']


def test_read_only_ui_preserves_unsaved_and_saved_edits(tmp_path,monkeypatch):
    from trustquery.review_store import ReviewStore,load_batch
    monkeypatch.chdir(tmp_path)
    batch=tmp_path/'data/batch_runs/major_deliverable_1/results.json'
    batch.parent.mkdir(parents=True)
    r=row()
    batch.write_text(json.dumps([r]))
    db=tmp_path/'data/reviews/major_deliverable_2.sqlite3'
    store=ReviewStore(db,load_batch(batch)[1])
    app=local_app(AppTest, str(ROOT/'src/trustquery/review_app.py'),default_timeout=40).run()
    assert not app.exception
    assert not any(b.label=='Locate full policy sections (read only)' for b in app.button)
    element(app,'text_area','Answer draft').set_value('My unsaved review').run()
    element(app,'toggle','Enable read-only section candidates').set_value(True).run()
    element(app,'button','Locate full policy sections (read only)').click().run()
    assert not app.exception
    assert element(app,'text_area','Answer draft').value=='My unsaved review'
    assert store.current(r)['revision']==0
    assert any('UNVERIFIED SECTION' in e.label for e in app.expander)
    element(app,'text_input','Reviewer name').set_value('Test reviewer').run()
    element(app,'button','Save edit').click().run()
    saved=store.current(r)
    element(app,'button','Locate full policy sections (read only)').click().run()
    assert store.current(r)==saved and saved['answer']=='My unsaved review'
    assert saved['status']=='edited' and saved['citations']==[]
    # New locators never enter export; existing candidate/verified separation remains.
    from trustquery.review_export import export_bytes
    from test_review_export import read
    _,verified,candidates,_=read(export_bytes(batch,db))
    assert len(verified)==1 and len(candidates)==1
