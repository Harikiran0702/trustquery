"""Isolated offline review store. Candidate locators never become final citations."""
import json
from pathlib import Path
import sqlite3

from trustquery.file_hashes import legacy_crlf_sha256, text_sha256


class ReviewConflict(ValueError):
    """The submitted displayed revision is no longer current."""


def load_batch(path, *, include_legacy_hash=False):
    raw = Path(path).read_bytes()
    rows = json.loads(raw)
    if not isinstance(rows, list) or not rows:
        raise ValueError('Expected a nonempty batch result array')
    ids = set()
    for r in rows:
        if not isinstance(r, dict) or not isinstance(r.get('id'), str) or not r['id'].strip() or r['id'] in ids:
            raise ValueError('Missing or duplicate question ID')
        ids.add(r['id'])
        if not isinstance(r.get('metadata', {}), dict) or not isinstance(r.get('metadata', {}).get('domain', ''), str) or not isinstance(r.get('outcome'), str) or not isinstance(r.get('evidence_assessment'), dict):
            raise ValueError('Invalid batch metadata, outcome or assessment')
        if not isinstance(r.get('question'), str) or not isinstance(r.get('answer'), str) or not isinstance(r.get('evidence_references'), list):
            raise ValueError('Expected major deliverable 1 results with answer and evidence_references')
        for e in r['evidence_references']:
            if not isinstance(e, dict) or any(not isinstance(e.get(k), str) or not e[k].strip() for k in ('source', 'chunk_id', 'quote')):
                raise ValueError('Invalid candidate reference')
            if type(e.get('page')) is not int or e['page'] < 1 or type(e.get('chunk_index')) is not int or e['chunk_index'] < 0:
                raise ValueError('Invalid candidate locator')
    digest = text_sha256(raw)
    if include_legacy_hash:
        return rows, digest, legacy_crlf_sha256(raw)
    return rows, digest


def policy_extract(record, selected):
    refs = {e['chunk_id']: e for e in record['evidence_references']}
    if not selected or len(selected) != len(set(selected)) or any(k not in refs for k in selected):
        raise ValueError('Select unique candidate passages')
    return ('The retrieved synthetic policy text states:\n\n' +
            '\n\n'.join('"' + refs[k]['quote'] + '"' for k in selected) +
            '\n\nPolicy adoption and implemented controls are not established.')


class ReviewStore:
    def __init__(self, path, batch_hash, *, legacy_hash=None):
        self.path = Path(path)
        self.batch_hash = batch_hash
        if self.path.suffix.lower() != '.sqlite3':
            raise ValueError('Review state must use a separate .sqlite3 file')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute('CREATE TABLE IF NOT EXISTS batch (hash TEXT NOT NULL)')
            hashes = db.execute('SELECT hash FROM batch').fetchall()
            if hashes and hashes != [(batch_hash,)] and (legacy_hash is None or hashes != [(legacy_hash,)]):
                raise ValueError('Review database belongs to a different batch; choose a new database')
            if not hashes:
                db.execute('INSERT INTO batch VALUES (?)', (batch_hash,))
            db.execute('CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY, question TEXT NOT NULL, revision INTEGER NOT NULL, payload TEXT NOT NULL, created TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, UNIQUE(question, revision))')

    def connect(self):
        return sqlite3.connect(self.path, timeout=10)

    def current(self, record):
        with self.connect() as db:
            row = db.execute('SELECT payload FROM events WHERE question=? ORDER BY revision DESC LIMIT 1', (record['id'],)).fetchone()
        return json.loads(row[0]) if row else dict(revision=0, status='unreviewed', answer=record['answer'], selected=[], reviewer='', note='', scope_checked=False, citations=[])

    def history(self, question):
        with self.connect() as db:
            return [dict(timestamp=t, **json.loads(p)) for t, p in db.execute('SELECT created,payload FROM events WHERE question=? ORDER BY revision', (question,))]

    def save(self, record, *, revision, action, answer, selected, reviewer, note, scope_checked=False, experimental_policy=None):
        if action not in ('edited', 'approved', 'rejected', 'needs_evidence'):
            raise ValueError('Unknown review action')
        if not reviewer.strip():
            raise ValueError('Reviewer name is required')
        refs = {e['chunk_id'] for e in record['evidence_references']}
        if any(k not in refs for k in selected):
            raise ValueError('Unknown candidate reference')
        if action in ('rejected', 'needs_evidence') and not note.strip():
            raise ValueError('Record the rejection reason or missing evidence')
        if experimental_policy is not None:
            from trustquery.review_policy import validate
            validate(record, experimental_policy, answer, approve=action == 'approved')
        if action == 'approved':
            assessment = record.get('evidence_assessment', {})
            if assessment.get('implementation_required') is not False or record.get('outcome') in ('processing_error', 'implementation_evidence_required'):
                raise ValueError('Operational or failed records require corroborating implementation evidence; approval is blocked')
            if not scope_checked or not note.strip():
                raise ValueError('Record a scope/completeness review and rationale')
            if experimental_policy is None and answer != policy_extract(record, selected):
                raise ValueError('Only the exact bounded synthetic policy extract can be approved; edited claims require further evidence validation')
        payload = dict(revision=revision + 1, status=action, answer=answer, selected=list(selected), reviewer=reviewer.strip(), note=note.strip(), scope_checked=bool(scope_checked), citations=[], operational_proof=False, approval_scope='synthetic_policy_extract_only' if action == 'approved' else None)
        if experimental_policy is not None:
            payload['experimental_policy'] = experimental_policy
            payload['approval_scope'] = 'experimental_proposed_policy_extraction_only' if action == 'approved' else None
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            actual = db.execute('SELECT COALESCE(MAX(revision),0) FROM events WHERE question=?', (record['id'],)).fetchone()[0]
            if actual != revision:
                raise ReviewConflict('Review changed in another session. Compare before saving')
            db.execute('INSERT INTO events(question,revision,payload) VALUES (?,?,?)', (record['id'], revision + 1, json.dumps(payload, sort_keys=True)))
        return payload
