"""Isolated blinded collection. Never imports generation or comparison modules."""
import argparse
import hashlib
import json
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from http.server import BaseHTTPRequestHandler

LABELS = {'supported', 'contradicted', 'insufficient', 'implementation-evidence-required', 'defer'}
JUDGMENTS = ('scope', 'population', 'modality', 'polarity', 'exceptions', 'approval_status', 'conflicts', 'components', 'citation_judgment')
PACKET_SHA = '2c26d5a6e8a746fe5c182bf34fe35b8ac1db5d9b67a82fbbeeb4bb252ec60ca7'
def digest(b): return hashlib.sha256(b).hexdigest()
def encoded(v): return json.dumps(v, sort_keys=True, ensure_ascii=False).encode()
def now(): return datetime.now(timezone.utc).isoformat()
def require(ok, message):
    if not ok:
        raise ValueError(message)

class Collection:
    def __init__(self, db, packet, pdfs, manifest):
        self.packet_path, self.pdfs, self.manifest = Path(packet), Path(pdfs), Path(manifest)
        self.packet = json.loads(self.packet_path.read_bytes())
        self.verify_sources()
        self.cases = {c['case_id']: c for c in self.packet['cases']}
        require(len(self.cases) == len(self.packet['cases']), 'duplicate cases')
        self.evidence = {f"{e['source']}:p{e['page']}": e for e in self.packet['evidence']}
        require(len(self.evidence)==len(self.packet['evidence']), 'duplicate evidence')
        self.db = sqlite3.connect(db, isolation_level=None, check_same_thread=False)
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS reviewers(identity TEXT PRIMARY KEY, token TEXT UNIQUE, role TEXT, attestation TEXT, registered_at TEXT);
        CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, case_id TEXT, reviewer TEXT, kind TEXT, submitted_at TEXT, payload TEXT, previous TEXT, hash TEXT);
        CREATE TRIGGER IF NOT EXISTS immutable_update BEFORE UPDATE ON events BEGIN SELECT RAISE(ABORT,'append only'); END;
        CREATE TRIGGER IF NOT EXISTS immutable_delete BEFORE DELETE ON events BEGIN SELECT RAISE(ABORT,'append only'); END;
        CREATE TRIGGER IF NOT EXISTS identity_update BEFORE UPDATE ON reviewers BEGIN SELECT RAISE(ABORT,'immutable identity'); END;
        CREATE TRIGGER IF NOT EXISTS identity_delete BEFORE DELETE ON reviewers BEGIN SELECT RAISE(ABORT,'immutable identity'); END;
        ''')
    def verify_sources(self):
        require(digest(self.packet_path.read_bytes()) == PACKET_SHA, 'packet version mismatch')
        manifest = json.loads(self.manifest.read_bytes())
        require(manifest == self.packet['corpus_sha256'], 'trusted manifest mismatch')
        for source, expected in manifest.items():
            require(Path(source).name == source, 'unsafe source')
            require(digest((self.pdfs/source).read_bytes()) == expected, 'original PDF hash mismatch: '+source)
        for e in self.packet['evidence']:
            require(e['pdf_sha256'] == manifest[e['source']], 'source version mismatch')
            require(digest(e['text'].encode()) == e['page_text_sha256'], 'page text mismatch')
    def register(self, identity, role, attestation):
        require(identity.strip() == identity and bool(identity), 'explicit identity required')
        require(role in ('initial', 'adjudicator', 'internal'), 'invalid role')
        require(attestation.strip(), 'custodian identity verification and independence attestation required')
        if role == 'initial':
            require(self.db.execute("SELECT count(*) FROM reviewers WHERE role='initial'").fetchone()[0] < 2, 'two initial slots already assigned')
        token = secrets.token_urlsafe(32)
        self.db.execute('INSERT INTO reviewers VALUES (?,?,?,?,?)', (identity,digest(token.encode()),role,attestation,now()))
        return token
    def actor(self, token):
        row = self.db.execute('SELECT identity,role FROM reviewers WHERE token=?',(digest(token.encode()),)).fetchone()
        require(row is not None, 'invalid credential')
        return row
    def events(self):
        out=[]
        previous=''
        for row in self.db.execute('SELECT * FROM events ORDER BY id'):
            _,case,reviewer,kind,at,payload,prev,h=row
            data=dict(case_id=case,reviewer=reviewer,kind=kind,submitted_at=at,payload=json.loads(payload),previous=prev)
            require(prev==previous and digest(encoded(data))==h, 'audit chain mismatch')
            datetime.fromisoformat(at)
            out.append(dict(data,id=row[0],hash=h))
            previous=h
        return out
    def validate_review(self, p):
        require(set(p)=={'label','rationale','bindings','judgments','independent','exposure','reason','supersedes'}, 'unexpected or missing fields; timestamps and identities are server-owned')
        require(p['label'] in LABELS and isinstance(p['rationale'],str) and p['rationale'].strip(), 'label/rationale required')
        require(p['independent'] is True and p['exposure']=='none', 'exposed/nonindependent review cannot qualify')
        require(set(p['judgments'])==set(JUDGMENTS) and all(isinstance(v,str) and v.strip() for v in p['judgments'].values()), 'complete scope/citation judgments required')
        require(isinstance(p['bindings'],list) and p['bindings'], 'exact evidence bindings required even for absence/defer')
        seen=set()
        for b in p['bindings']:
            require(set(b)=={'evidence_id','source','page','section','pdf_sha256','page_text_sha256','quote'}, 'incomplete evidence binding')
            eid=b['evidence_id']
            require(eid in self.evidence and eid not in seen,'unknown/duplicate evidence')
            seen.add(eid)
            e=self.evidence[eid]
            require(all(b[k]==e[k] for k in ('source','page','section','pdf_sha256','page_text_sha256')), 'evidence/source version mismatch')
            require(isinstance(b['quote'],str) and b['quote'].strip() and b['quote'] in e['text'],'quote must be exact')
    def view(self, token, case):
        identity,role=self.actor(token)
        require(case in self.cases,'unknown case')
        events=[e for e in self.events() if e['case_id']==case]
        both=len({e['reviewer'] for e in events if e['kind']=='initial'})==2
        return dict(case=self.cases[case], evidence=self.packet['evidence'], reviews=[e for e in events if e['reviewer']==identity or (both and role=='adjudicator')])
    def submit(self, token, case, kind, p):
        identity,role=self.actor(token)
        require(case in self.cases,'unknown case')
        self.verify_sources()
        self.validate_review(p)
        require(kind in ('initial','correction','resolution','internal'),'invalid event kind')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            all_events=self.events()
            es=[e for e in all_events if e['case_id']==case]
            initial=[e for e in es if e['kind']=='initial']
            if kind=='initial':
                require(role=='initial' and not any(e['reviewer']==identity for e in initial),'duplicate initial/incorrect role')
                require(p['supersedes'] is None,'initial cannot supersede')
            elif kind=='correction':
                require(p['reason'].strip(),'correction reason required')
                require(any(e['id']==p['supersedes'] and e['reviewer']==identity and e['kind'] in ('initial','correction') for e in es),'invalid correction parent')
                own=[e for e in es if e['reviewer']==identity and e['kind'] in ('initial','correction')]
                require(own[-1]['id']==p['supersedes'],'stale correction parent')
            elif kind=='resolution':
                require(role=='adjudicator' and len(initial)==2 and identity not in {e['reviewer'] for e in initial},'third distinct reviewer required after two submissions')
                require(p['reason'].strip() and p['supersedes'] is None,'resolution rationale required')
            else:
                require(role=='internal','internal role required')
            data=dict(case_id=case,reviewer=identity,kind=kind,submitted_at=now(),payload=p,previous=all_events[-1]['hash'] if all_events else '')
            h=digest(encoded(data))
            self.db.execute('INSERT INTO events(case_id,reviewer,kind,submitted_at,payload,previous,hash) VALUES(?,?,?,?,?,?,?)',(case,identity,kind,data['submitted_at'],json.dumps(p),data['previous'],h))
            self.db.execute('COMMIT')
            return h
        except Exception:
            self.db.execute('ROLLBACK')
            raise
    def readiness(self):
        self.verify_sources()
        events=self.events()
        blockers=[]
        registered={r[0] for r in self.db.execute("SELECT identity FROM reviewers WHERE role='initial'")}
        if len(registered)!=2:
            blockers.append('exactly two verified initial reviewers required')
        for case in self.cases:
            es=[e for e in events if e['case_id']==case]
            initial=[e for e in es if e['kind']=='initial']
            if len(initial)!=2 or {e['reviewer'] for e in initial}!=registered:
                blockers.append(case+': two initial reviews missing')
                continue
            latest={e['reviewer']:e for e in initial}
            for e in es:
                self.validate_review(e['payload'])
                if e['kind']=='correction':
                    latest[e['reviewer']]=e
            labels={e['payload']['label'] for e in latest.values()}
            resolutions=[e for e in es if e['kind']=='resolution']
            if 'defer' in labels or len(labels)!=1 or case=='BSR023' or any(e['kind']=='correction' for e in es):
                if not resolutions or resolutions[-1]['id'] < max(e['id'] for e in latest.values()) or resolutions[-1]['payload']['label']=='defer':
                    blockers.append(case+': third-step resolution required (including historical BSR023 alternative)')
        return dict(ready=not blockers, blockers=blockers, event_count=len(events), seal_created=False, predictions_opened=False)

def serve(c, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def do_POST(self):
            try:
                token=self.headers.get('Authorization','').removeprefix('Bearer ')
                require(int(self.headers.get('Content-Length','0')) <= 1000000,'request too large')
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if self.path=='/case':
                    result=c.view(token,body['case_id'])
                elif self.path=='/submit':
                    result={'receipt':c.submit(token,body['case_id'],body['kind'],body['review'])}
                else:
                    raise ValueError('unknown endpoint')
                data=encoded(result)
                self.send_response(200)
            except (ValueError,KeyError,sqlite3.Error) as exc:
                data=encoded({'error':str(exc)})
                self.send_response(400)
            self.send_header('Content-Type','application/json')
            self.send_header('Cache-Control','no-store')
            self.end_headers()
            self.wfile.write(data)
    # Serial request processing prevents transaction interleaving on the shared connection.
    from http.server import HTTPServer
    HTTPServer(('127.0.0.1',port),Handler).serve_forever()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('db','packet','pdfs','manifest'):
        p.add_argument('--'+name,required=True)
    sub=p.add_subparsers(dest='command',required=True)
    r=sub.add_parser('register')
    r.add_argument('--identity',required=True)
    r.add_argument('--role',choices=['initial','adjudicator','internal'],required=True)
    r.add_argument('--attestation',required=True)
    sub.add_parser('validate')
    s=sub.add_parser('serve')
    s.add_argument('--port',type=int,default=8765)
    a=p.parse_args()
    c=Collection(a.db,a.packet,a.pdfs,a.manifest)
    if a.command=='register':
        print(c.register(a.identity,a.role,a.attestation))
    elif a.command=='validate':
        result=c.readiness()
        print(json.dumps(result,indent=2))
        raise SystemExit(0 if result['ready'] else 1)
    else:
        serve(c,a.port)
if __name__=='__main__':
    main()
