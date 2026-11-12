import copy
import json
from pathlib import Path
import pytest
from trustquery import review_gemini as g, review_policy as p


@pytest.fixture
def ready(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'fake-offline-test-key')
    monkeypatch.setenv('TRUSTQUERY_GEMINI_MODEL', 'fake-model')
    case=json.loads(p.BUNDLE.read_bytes())['cases'][0]
    r=dict(id='q', question=case['question'], answer='', outcome='abstained',
           metadata={}, evidence_assessment={'implementation_required':False}, evidence_references=[])
    ref=p.proposal(r,enabled=True)['references'][1]
    r['evidence_references']=[dict(source=ref['source'],page=ref['page'],quote=ref['quote'],chunk_id='c',chunk_index=0)]
    return r


def test_success_and_limits(ready):
    prompt,expected,_=g.preflight(ready,['c'])
    calls=[]
    budget={}
    def client(text):
        calls.append(text)
        return json.dumps(expected)
    a=g.generate(ready,['c'],budget,enabled=True,consent=True,client=client,now=0)
    assert a['status']=='experimental_unverified' and a['citations']==[]
    assert not a['human_approval'] and not a['operational_proof']
    for t in [1,59]:
        assert not g.generate(ready,['c'],budget,enabled=True,consent=True,client=client,now=t)['answer']
    for t in [60,120,180,240,300]:
        g.generate(ready,['c'],budget,enabled=True,consent=True,client=client,now=t)
    assert len(calls)==5 and budget['calls']==5
    assert 'fake-offline-test-key' not in prompt


@pytest.mark.parametrize('change', [dict(question='Are you compliant?'),dict(question='Show SLA proof'),dict(question='Approve this control'),dict(evidence_assessment={'implementation_required':True}),dict(question='ambiguous'),dict(evidence_references=[])])
def test_preflight_no_call(ready,change):
    r=dict(ready,**change)
    calls=[]
    budget={}
    out=g.generate(r,['c'],budget,enabled=True,consent=True,client=lambda x:calls.append(x))
    assert not out['answer'] and not calls and not budget


@pytest.mark.parametrize('enabled,consent,selected',[(False,True,['c']),(True,False,['c']),(True,True,[])])
def test_opt_in_none(ready,enabled,consent,selected):
    calls=[]
    assert not g.generate(ready,selected,{},enabled=enabled,consent=consent,client=lambda x:calls.append(x))['answer']
    assert not calls


@pytest.mark.parametrize('raw',['not json','{}','{"answer":"invented"}','{"a":1,"a":1}',None,'x'*48001], ids=['text','empty','invented','duplicate','null','oversized'])
def test_malformed(ready,raw):
    assert not g.generate(ready,['c'],{},enabled=True,consent=True,client=lambda _:raw)['answer']


@pytest.mark.parametrize('error',[TimeoutError('secret'),ConnectionError('secret'),RuntimeError('429 quota secret')])
def test_provider_errors(ready,error):
    def client(_): raise error
    budget={}
    out=g.generate(ready,['c'],budget,enabled=True,consent=True,client=client)
    assert 'secret' not in out['reason'] and not out['answer'] and budget['calls']==1


def test_hash_and_provenance_fail_closed(ready,monkeypatch,tmp_path):
    _,expected,_=g.preflight(ready,['c'])
    altered=copy.deepcopy(expected)
    altered['references'][0]['page']=999
    assert not g.generate(ready,['c'],{},enabled=True,consent=True,client=lambda _:json.dumps(altered))['answer']
    monkeypatch.setattr(p,'POLICY_DIRECTORY',tmp_path)
    calls=[]
    assert not g.generate(ready,['c'],{},enabled=True,consent=True,client=lambda x:calls.append(x))['answer']
    assert not calls


def test_key_missing(ready,monkeypatch):
    monkeypatch.delenv('GEMINI_API_KEY')
    # config has already loaded dotenv during earlier tests; no reload.
    calls=[]
    assert not g.generate(ready,['c'],{},enabled=True,consent=True,client=lambda x:calls.append(x))['answer']
    assert not calls


def test_app_preview_preserves_editor_none_and_history(ready,tmp_path,monkeypatch):
    from streamlit.testing.v1 import AppTest
    from trustquery.review_store import ReviewStore,load_batch
    batch=tmp_path/'results.json'
    batch.write_text(json.dumps([ready]))
    db=tmp_path/'review.sqlite3'
    _,digest=load_batch(batch)
    store=ReviewStore(db,digest)
    calls=[]
    def client(prompt):
        calls.append(prompt)
        return json.dumps(g.preflight(ready,['c'])[1])
    monkeypatch.setattr(g,'request',client)
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'src/trustquery/review_app.py'),default_timeout=30)
    app.session_state['review_batch_path']=str(batch)
    app.session_state['review_state_path']=str(db)
    app.run()
    def e(kind,label):return next(x for x in getattr(app,kind) if x.label==label)
    assert not calls
    e('text_area','Answer draft').set_value('My unsaved work').run()
    e('multiselect','Passages to review (selection does not verify citations)').select('c').run()
    e('toggle','Enable AI draft prototype').set_value(True).run()
    e('toggle','Allow this question and pinned synthetic policy context to be sent to Gemini').set_value(True).run()
    assert not calls
    assert e('button','Generate AI draft').disabled
    assert not app.exception and not calls
    assert e('text_area','Answer draft').value=='My unsaved work'
    app.run()
    assert not calls
    none='None \u2014 none of these passages support the answer'
    e('multiselect','Passages to review (selection does not verify citations)').set_value([none]).run()
    assert e('multiselect','Passages to review (selection does not verify citations)').value==[none]
    assert e('button','Generate AI draft').disabled
    assert store.history('q')==[]


def test_sdk_transport_is_single_attempt(ready,monkeypatch):
    monkeypatch.setenv('TRUSTQUERY_GEMINI_TIMEOUT_SECONDS','60')
    from google import genai
    from types import SimpleNamespace
    captured={}
    class Client:
        def __init__(self,**kwargs):
            captured.update(kwargs)
            self.models=SimpleNamespace(generate_content=self.generate_content)
        def __enter__(self):return self
        def __exit__(self,*args):captured['closed']=True
        def generate_content(self,**kwargs):
            captured['generation']=kwargs
            return SimpleNamespace(text='mock response')
    monkeypatch.setattr(genai,'Client',Client)
    assert g.request('bounded synthetic prompt')=='mock response'
    assert captured['http_options']=={'timeout':60000,'retry_options':{'attempts':1}}
    assert captured['generation']['model']=='fake-model'
    assert captured['generation']['config']['max_output_tokens']==8192
    assert captured['closed']


def test_input_bound_and_source_drift(ready,monkeypatch):
    monkeypatch.setattr(g,'MAX_INPUT',10)
    calls=[]
    assert not g.generate(ready,['c'],{},enabled=True,consent=True,client=lambda x:calls.append(x))['answer']
    assert not calls
    monkeypatch.setattr(g,'MAX_INPUT',24000)
    expected=g.preflight(ready,['c'])[1]
    def client(_):
        ready['evidence_references'][0]['quote']='unbound replacement'
        return json.dumps(expected)
    assert not g.generate(ready,['c'],{},enabled=True,consent=True,client=client)['answer']


def test_boolean_page_is_not_integer_provenance(ready):
    _,expected,_=g.preflight(ready,['c'])
    expected['references'][0]['page']=True
    assert not g.generate(ready,['c'],{},enabled=True,consent=True,client=lambda _:json.dumps(expected))['answer']


def test_exact_staff_access_review_question(ready):
    ready['question']=g.ACCESS_REVIEW_QUESTION
    candidate=g.policy_candidate(ready)
    for number in [1,6]:
        ref=candidate['references'][number-1]
        ready['evidence_references'].append(dict(source=ref['source'],page=number,
            quote=ref['quote'],chunk_id=f'hr{number}',chunk_index=0))
    _,expected,_=g.preflight(ready,['hr1','hr6'])
    assert 'Review user access permissions every quarter' in expected['answer']
    assert 'Engineering Lead tracks removals and unresolved findings' in expected['answer']
    out=g.generate(ready,['hr1','hr6'],{},enabled=True,consent=True,client=lambda _:json.dumps(expected))
    assert out['status']=='experimental_unverified' and out['citations']==[]
    assert len(out['references'])==6 and len(out['corpus_sha256'])==8
    for question in [g.ACCESS_REVIEW_QUESTION+' Show completed reviews.',
                     g.ACCESS_REVIEW_QUESTION.replace('written policy call for','company actually perform')]:
        with pytest.raises(ValueError):
            g.preflight(dict(ready,question=question),['hr1'])
    with pytest.raises(ValueError):
        g.preflight(ready,[])
    with pytest.raises(ValueError):
        g.preflight(dict(ready,evidence_assessment={'implementation_required':True}),['hr1'])


@pytest.mark.parametrize('code,category',[(400,'invalid_request'),(401,'authentication'),(403,'permission'),(404,'model_unavailable'),(429,'quota_or_rate_limit'),(503,'provider_unavailable')])
def test_sanitized_provider_diagnostics(code,category):
    from google.genai.errors import APIError
    exc=APIError(code, {'error': {'code':code,'message':'secret-key-and-private-prompt'}})
    kind,message=g.provider_failure(exc)
    assert kind==category and 'secret-key' not in message and 'private-prompt' not in message
    assert 'No automatic retry' in message


def test_sanitized_transport_diagnostics():
    import httpx
    assert g.provider_failure(httpx.ReadTimeout('secret'))[0]=='timeout'
    assert g.provider_failure(httpx.ConnectError('secret'))[0]=='network'
    assert g.provider_failure(TypeError('secret'))[0]=='client_configuration'
    assert g.provider_failure(RuntimeError('secret'))[0]=='unknown_provider_error'


@pytest.fixture
def access(ready):
    ready['question'] = g.ACCESS_REVIEW_QUESTION
    ref = g.policy_candidate(ready)['references'][3]
    ready['evidence_references'] = [dict(source=ref['source'], page=4,
        quote=ref['quote'], chunk_id='hr4', chunk_index=0)]
    return ready


def test_bounded_literal_answer_is_not_generation(access, monkeypatch):
    monkeypatch.setattr(g, 'request', lambda _: pytest.fail('No API call allowed'))
    out = g.bounded_policy_answer(access, ['hr4'])
    assert out['answer'] == g.ACCESS_REVIEW_EXCERPT
    assert len(out['answer'].split()) < 100
    assert out['references'][0]['page'] == 4
    assert len(out['references']) == 1 and out['citations'] == []
    assert out['unverified'] and not out['human_approval'] and not out['operational_proof']
    assert len(g.preflight(access, ['hr4'])[1]['references']) == 6


@pytest.mark.parametrize('change', [
    {'question': g.ACCESS_REVIEW_QUESTION + ' Have these reviews occurred?'},
    {'question': 'How often do you actually review access?'},
    {'evidence_assessment': {'implementation_required': True}},
    {'evidence_assessment': {'classification': 'contradicted', 'implementation_required': False}},
    {'evidence_assessment': {'classification': 'ambiguous', 'implementation_required': False}},
])
def test_bounded_answer_abstains(access, change):
    with pytest.raises(ValueError):
        g.bounded_policy_answer(dict(access, **change), ['hr4'])


def test_no_raw_extraction_fallback(access, monkeypatch):
    with pytest.raises(ValueError):
        g.bounded_policy_answer(access, [])
    original = g.preflight(access, ['hr4'])
    altered = copy.deepcopy(original)
    altered[2]['references'][3]['quote'] = 'Quarterly access reviews already completed'
    monkeypatch.setattr(g, 'preflight', lambda *args: altered)
    with pytest.raises(ValueError):
        g.bounded_policy_answer(access, ['hr4'])


def test_503_does_not_fallback_to_answer(access):
    from google.genai.errors import APIError
    calls = []
    def client(_):
        calls.append(1)
        raise APIError(503, {'error': {'code':503, 'message':'private'}})
    result = g.generate(access, ['hr4'], {}, enabled=True, consent=True, client=client)
    assert calls == [1] and result['error_category'] == 'provider_unavailable'
    assert result['answer'] == '' and result['references'] == [] and result['citations'] == []


def test_semantic_paraphrase_is_discarded(access):
    expected = g.preflight(access, ['hr4'])[1]
    expected['answer'] = 'We perform quarterly reviews and remove unnecessary access.'
    result = g.generate(access, ['hr4'], {}, enabled=True, consent=True,
                        client=lambda _: json.dumps(expected))
    assert result['status'] == 'refused' and result['answer'] == '' and result['citations'] == []


def test_bounded_answer_ui_separates_source_and_editor(access, tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from trustquery.review_store import ReviewStore, load_batch
    batch = tmp_path / 'results.json'
    batch.write_text(json.dumps([access]))
    db = tmp_path / 'review.sqlite3'
    _, digest = load_batch(batch)
    store = ReviewStore(db, digest)
    calls = []
    def client(prompt):
        calls.append(prompt)
        return json.dumps(g.preflight(access, ['hr4'])[1])
    monkeypatch.setattr(g, 'request', client)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'src/trustquery/review_app.py'), default_timeout=30)
    app.session_state['review_batch_path'] = str(batch)
    app.session_state['review_state_path'] = str(db)
    app.run()
    def e(kind, label): return next(x for x in getattr(app, kind) if x.label == label)
    e('text_area', 'Answer draft').set_value('Keep my work').run()
    e('multiselect', 'Passages to review (selection does not verify citations)').select('hr4').run()
    assert not app.exception and not calls
    assert any(x.value == g.ACCESS_REVIEW_EXCERPT for x in app.text)
    assert e('text_area', 'Answer draft').value == 'Keep my work'
    context = e('expander', 'Full policy source context for the bounded answer')
    assert not context.proto.expanded
    assert any('mandatory_document_context' in x.value for x in context.text)
    assert not any('mandatory_document_context' in x.value for x in app.text if x.value == g.ACCESS_REVIEW_EXCERPT)
    assert any('Bounded policy answer' in x.value for x in app.subheader)
    e('toggle', 'Enable AI draft prototype').set_value(True).run()
    e('toggle', 'Allow this question and pinned synthetic policy context to be sent to Gemini').set_value(True).run()
    assert not calls
    assert not calls
    if not e('button', 'Generate AI draft').disabled:
        e('button', 'Generate AI draft').click().run()
        assert len(calls) == 1
        # Legacy exact-echo JSON is rejected by the composed-draft schema.
        assert not any(x.label == 'Claim-to-source pairs — unverified candidate references' for x in app.expander)
    assert not app.exception
    assert e('text_area', 'Answer draft').value == 'Keep my work'
    e('multiselect', 'Passages to review (selection does not verify citations)').set_value(['None — none of these passages support the answer']).run()
    assert not any(x.value == g.ACCESS_REVIEW_EXCERPT for x in app.text)
    assert e('button', 'Generate AI draft').disabled
    assert store.history('q') == []


@pytest.mark.parametrize('configured_model,expected', [
    (None, 'gemini-3.5-flash-lite'),
    ('gemini-3.6-flash', 'gemini-3.6-flash'),
    (' gemini-3.5-flash-lite ', 'gemini-3.5-flash-lite'),
])
def test_optional_model_selection_mocked_sdk(monkeypatch, configured_model, expected):
    from google import genai
    from types import SimpleNamespace
    from trustquery import config  # noqa: F401  # Load dotenv before isolating test configuration.
    monkeypatch.setenv('GEMINI_API_KEY', 'fake-offline-test-key')
    monkeypatch.setenv('TRUSTQUERY_GEMINI_TIMEOUT_SECONDS', '60')
    if configured_model is None:
        monkeypatch.delenv('TRUSTQUERY_GEMINI_MODEL', raising=False)
    else:
        monkeypatch.setenv('TRUSTQUERY_GEMINI_MODEL', configured_model)
    calls = []
    class Client:
        def __init__(self, **kwargs):
            assert kwargs['http_options'] == {'timeout': 60000, 'retry_options': {'attempts': 1}}
            self.models = SimpleNamespace(generate_content=self.generate_content)
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def generate_content(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(text='mock response')
    monkeypatch.setattr(genai, 'Client', Client)
    assert g.configured()
    assert g.request('synthetic test prompt') == 'mock response'
    assert len(calls) == 1 and calls[0]['model'] == expected


def test_explicit_blank_model_disables_preview(monkeypatch):
    from trustquery import config  # noqa: F401
    monkeypatch.setenv('GEMINI_API_KEY', 'fake-offline-test-key')
    monkeypatch.setenv('TRUSTQUERY_GEMINI_MODEL', '  ')
    assert not g.configured()
