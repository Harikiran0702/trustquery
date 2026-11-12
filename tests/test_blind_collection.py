import copy
import hashlib
import json
import sqlite3
from pathlib import Path
import pytest
import trustquery.blind_collection as m

@pytest.fixture
def setup(tmp_path,monkeypatch):
    pdf=b'original bytes'
    h=hashlib.sha256(pdf).hexdigest()
    text='Exact evidence span.'
    packet={'cases':[{'case_id':'BSR023','question':'Question?'}], 'corpus_sha256':{'source.pdf':h}, 'evidence':[{'source':'source.pdf','page':1,'section':None,'pdf_sha256':h,'page_text_sha256':hashlib.sha256(text.encode()).hexdigest(),'text':text}]}
    path=tmp_path/'packet.json'
    path.write_text(json.dumps(packet))
    (tmp_path/'source.pdf').write_bytes(pdf)
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps(packet['corpus_sha256']))
    monkeypatch.setattr(m,'PACKET_SHA',m.digest(path.read_bytes()))
    c=m.Collection(tmp_path/'db',path,tmp_path,manifest)
    a=c.register('Real A','initial','Verified separately; no prior exposure')
    b=c.register('Real B','initial','Verified separately; no prior exposure')
    d=c.register('Real C','adjudicator','Verified third person')
    e=packet['evidence'][0]
    binding={k:v for k,v in e.items() if k!='text'}
    binding.update(evidence_id='source.pdf:p1',quote=text)
    p=dict(label='supported',rationale='Human rationale',bindings=[binding],judgments={k:'Human judgment' for k in m.JUDGMENTS},independent=True,exposure='none',reason='',supersedes=None)
    return c,a,b,d,p

def test_isolation_identity_time_and_duplicates(setup,monkeypatch):
    c,a,b,d,p=setup
    monkeypatch.setattr(m,'now',lambda:'2026-10-31T02:30:00+00:00')
    c.submit(a,'BSR023','initial',p)
    assert c.view(b,'BSR023')['reviews']==[]
    assert c.view(d,'BSR023')['reviews']==[]
    e=c.events()[0]
    assert e['reviewer']=='Real A' and e['submitted_at']=='2026-10-31T02:30:00+00:00'
    with pytest.raises(ValueError):
        c.submit(a,'BSR023','initial',p)
    with pytest.raises(ValueError):
        c.view('bad','BSR023')
    c.submit(b,'BSR023','initial',p)
    assert len(c.view(d,'BSR023')['reviews'])==2
    assert len(c.view(a,'BSR023')['reviews'])==1
    with pytest.raises(sqlite3.IntegrityError):
        c.db.execute('DELETE FROM events')

@pytest.mark.parametrize('change',[{'submitted_at':'fake'},{'bindings':[]},{'exposure':'familiar'},{'judgments':{}},{'independent':False}])
def test_required_fields(setup,change):
    c,a,b,d,p=setup
    p.update(change)
    with pytest.raises(ValueError):
        c.submit(a,'BSR023','initial',p)

def test_hash_mismatch(setup):
    c,a,b,d,p=setup
    p['bindings'][0]['pdf_sha256']='bad'
    with pytest.raises(ValueError):
        c.submit(a,'BSR023','initial',p)
    (c.pdfs/'source.pdf').write_bytes(b'changed')
    with pytest.raises(ValueError):
        c.readiness()

def test_resolution_correction_and_gates(setup):
    c,a,b,d,p=setup
    assert not c.readiness()['ready']
    with pytest.raises(ValueError):
        c.submit(d,'BSR023','resolution',p)
    c.submit(a,'BSR023','initial',p)
    q=copy.deepcopy(p)
    q['label']='contradicted'
    c.submit(b,'BSR023','initial',q)
    assert not c.readiness()['ready']
    q['reason']='Third reviewer explains disagreement and BSR023 historical alternative'
    c.submit(d,'BSR023','resolution',q)
    assert c.readiness()['ready']
    q['supersedes']=1
    q['reason']='Correct initial review'
    c.submit(a,'BSR023','correction',q)
    assert not c.readiness()['ready']
    with pytest.raises(ValueError):
        c.submit(a,'BSR023','correction',q)
    q['supersedes']=None
    c.submit(d,'BSR023','resolution',q)
    assert c.readiness()['ready']

def test_internal_never_qualifies(setup):
    c,a,b,d,p=setup
    t=c.register('Developer','internal','Prior familiarity disclosed')
    c.submit(t,'BSR023','internal',p)
    assert not c.readiness()['ready']

def test_no_prediction_imports():
    source=Path(m.__file__).read_text()
    assert 'section_blind_eval' not in source and 'predictions.private' not in source

def test_http_credential_and_peer_isolation(setup):
    import threading
    import urllib.request
    import urllib.error
    from http.server import HTTPServer, BaseHTTPRequestHandler
    # Exercise reviewer CLI transport against a local mock with real Collection authorization.
    c,a,b,d,p=setup
    c.submit(a,'BSR023','initial',p)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def do_POST(self):
            body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            try:
                result=c.view(self.headers['Authorization'].removeprefix('Bearer '),body['case_id'])
                status=200
            except ValueError:
                result={'error':'denied'}
                status=403
            self.send_response(status)
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
    server=HTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    try:
        req=urllib.request.Request(f'http://127.0.0.1:{server.server_port}/case',data=b'{"case_id":"BSR023"}',headers={'Authorization':'Bearer '+b})
        with urllib.request.urlopen(req) as r:
            assert json.load(r)['reviews']==[]
        req.headers['Authorization']='Bearer invalid'
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(req)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

def test_client_owned_times_rejected_and_chain_detected(setup):
    c,a,b,d,p=setup
    q=copy.deepcopy(p)
    q['reviewer']='Someone else'
    with pytest.raises(ValueError):
        c.submit(a,'BSR023','initial',q)
    c.submit(a,'BSR023','initial',p)
    c.db.execute('DROP TRIGGER immutable_update')
    c.db.execute("UPDATE events SET submitted_at='2020-01-01T00:00:00+00:00'")
    with pytest.raises(ValueError,match='audit chain'):
        c.readiness()
