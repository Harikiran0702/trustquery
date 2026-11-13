"""Bounded, offline UI input adapter. Uploaded filenames never become paths."""
import hashlib
import json
from pathlib import Path
import tempfile
import uuid

from trustquery.batch import load_questionnaire
from trustquery.batch_process import process_questionnaire
from trustquery.batch_retrieval import load_corpus, POLICIES, offline_mode
from trustquery.review_store import ReviewStore, load_batch

ROOT = Path(__file__).resolve().parents[2]
MAX_BYTES = 1024 * 1024
MAX_QUESTIONS = 100


def validate_upload(name, raw):
    if (not isinstance(name, str) or not name or len(name) > 128
            or any(c in name for c in '/\\:') or name in ('.', '..')
            or any(ord(c) < 32 for c in name)):
        raise ValueError('Use a plain CSV or JSON filename without paths')
    suffix = Path(name).suffix.lower()
    if suffix not in ('.csv', '.json') or 'answer_key' in name.lower():
        raise ValueError('Only questionnaire CSV/JSON is supported; evaluation answer keys are forbidden')
    if not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_BYTES:
        raise ValueError('Questionnaire must contain 1 byte to 1 MiB')
    # The existing strict schema parser remains the single schema authority.
    with tempfile.TemporaryDirectory(prefix='trustquery-validate-') as tmp:
        path = Path(tmp) / ('questionnaire' + suffix)
        path.write_bytes(raw)
        questions, _ = load_questionnaire(path)
    if len(questions) > MAX_QUESTIONS:
        raise ValueError('At most 100 questions per upload')
    if any(len(q.id) > 128 or len(q.question) > 10000
           or len(json.dumps(q.metadata)) > 10000 for q in questions):
        raise ValueError('ID, question or metadata exceeds UI limits (128 / 10000 / 10000 characters)')
    for q in questions:
        if not isinstance(q.metadata.get('domain', ''), str):
            raise ValueError('metadata.domain must be a string')
        # Reject values which cannot be preserved in the review JSON/XLSX.
        text = json.dumps([q.id, q.question, q.metadata], ensure_ascii=False, allow_nan=False)
        if any(0xD800 <= ord(c) <= 0xDFFF for c in text):
            raise ValueError('Unpaired Unicode surrogates are unsupported')
        def check(value):
            if isinstance(value, str):
                if any(not (c in '\t\n\r' or 32 <= ord(c) <= 0xD7FF
                           or 0xE000 <= ord(c) <= 0xFFFD
                           or 0x10000 <= ord(c) <= 0x10FFFF) for c in value):
                    raise ValueError('XML-incompatible control characters are unsupported')
            elif isinstance(value, dict):
                for k, v in value.items():
                    check(k)
                    check(v)
            elif isinstance(value, list):
                for v in value:
                    check(v)
        check([q.id, q.question, q.metadata])
    return suffix, len(questions)


def corpus_hashes():
    folder = ROOT / 'data/sample'
    expected = json.loads((ROOT / 'src/trustquery/candidate_sections_manifest.json').read_text())['files']
    hashes = {}
    for name in POLICIES:
        path = folder / (name + '.pdf')
        if path.resolve() != path.absolute():
            raise ValueError('Redirected sample policies are unsupported')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected[path.name]:
            raise ValueError('Preloaded sample policy hash mismatch; processing stopped')
        hashes[path.name] = digest
    return hashes


def create_run(name, raw, directory=None, *, offline=False, hybrid=True, cache_ranking=True):
    payload, count = prepare_run(name, raw, offline=offline, hybrid=hybrid,
                                 cache_ranking=cache_ranking)
    return instantiate_run(payload, count, directory)


def prepare_run(name, raw, *, offline, hybrid, cache_ranking):
    suffix, count = validate_upload(name, raw)
    offline_mode()
    hashes = corpus_hashes()
    cache_chunks = load_corpus(ROOT / 'data/sample', preserve_layout=True)
    if corpus_hashes() != hashes:
        raise ValueError('Sample policies changed during ingestion')
    from trustquery import general_rag
    corpus = general_rag.load_corpus()
    chunks = corpus[0]
    retrieval_options = dict(corpus=corpus, hybrid=hybrid, cache_ranking=cache_ranking)
    retrieved = {}
    def retrieve(*, query, chunks, top_k, **unused):
        result = general_rag.retrieve(dict(question=query), **retrieval_options)
        retrieved[query] = result
        by_locator = {(s['metadata']['source'], s['metadata']['page']): s for s in chunks}
        return [dict(by_locator[c['source'], c['page']], **c['scores']) for c in result['candidates'][:top_k]]
    # No user path, executable, provider, index or model options are accepted.
    with tempfile.TemporaryDirectory(prefix='trustquery-process-') as tmp:
        source = Path(tmp) / ('questionnaire' + suffix)
        source.write_bytes(raw)
        records = process_questionnaire(source, chunks=chunks, collection=None,
            reranker_model=None, retrieve=retrieve, offline=offline, demo_cache_chunks=cache_chunks)
    if general_rag.load_corpus()[2] != corpus[2]:
        raise ValueError('Policy sources changed during batch retrieval')
    for r in records:
        result = retrieved.get(r['question'])
        if result is None:
            raise ValueError('Offline retrieval failed; no incomplete run saved')
        # Reapply final operational/ambiguity assessment to the saved eligibility.
        result = general_rag.retrieve(r, **retrieval_options)
        # Preserve conservative assessment; candidate availability never grants support.
        r['general_rag'] = result
        # Persist the complete preflight gates as well as retrieval availability.
        # Outgoing prompt/pages remain session-only; never store repeated context.
        prepared = general_rag.prepare(r, cache_ranking=cache_ranking)
        for field in ('eligible', 'reasons', 'disposition', 'reason_codes', 'disposition_reasons', 'next_action'):
            result[field] = prepared[field]
        r['provenance'].update(retrieval_method=result['method'],
            embedding_model='all-MiniLM-L6-v2' if result['fallback'] is None else None,
            reranker_model='ms-marco-MiniLM-L-6-v2' if result['fallback'] is None else None,
            ui_workflow=general_rag.VERSION, retrieval_fallback=result['fallback'],
            policy_pdf_sha256=hashes, policy_files_sha256=corpus[2])
        for ref in r['evidence_references']:
            candidate = next((c for c in result['candidates'] if c['source'] == ref['source'] and c['page'] == ref['page']), None)
            if candidate is not None:
                ref.update({k:candidate[k] for k in ('pdf_sha256','page_utf8_sha256','section_utf8_sha256','section_kind')})
    payload = json.dumps(records, ensure_ascii=False, allow_nan=False, indent=2).encode('utf-8')
    return payload, count


def instantiate_run(payload, count, directory):
    """Save initial batch results and initialize an empty review database."""
    parent = Path(directory) if directory is not None else run_directory()
    if parent.resolve() != parent.absolute():
        raise ValueError('Redirected UI run directory is unsupported')
    parent.mkdir(parents=True, exist_ok=True)
    run = parent / uuid.uuid4().hex
    run.mkdir()  # Exclusive allocation; repeated uploads never overwrite reviews.
    batch = run / 'results.json'
    with batch.open('xb') as stream:
        stream.write(payload)
    _, digest = load_batch(batch)
    db = run / 'review.sqlite3'
    ReviewStore(db, digest)
    return str(batch), str(db), count


def run_directory():
    return ROOT / 'data/reviews/ui_v1'


def reopen_run(run_id, directory=None):
    """Validate an existing UUID run and its binding without creating state."""
    import re
    from trustquery.review_export import snapshot
    if not isinstance(run_id, str) or not re.fullmatch(r'[0-9a-f]{32}', run_id):
        raise ValueError('Invalid saved run ID')
    parent = Path(directory) if directory is not None else run_directory()
    run = parent / run_id
    batch, db = run / 'results.json', run / 'review.sqlite3'
    for path in (parent, run, batch, db):
        if path.resolve() != path.absolute():
            raise ValueError('Redirected saved run paths are unsupported')
    if not batch.is_file() or not db.is_file():
        raise ValueError('Saved run requires existing results JSON and SQLite database')
    rows, digest, legacy_hash = load_batch(batch, include_legacy_hash=True)
    states = snapshot(db, digest, legacy_hash=legacy_hash)
    if set(states) - {row['id'] for row in rows}:
        raise ValueError('Review database contains unknown question IDs')
    return str(batch), str(db)


def saved_run_ids():
    return list_run_ids(run_directory())


def list_run_ids(directory):
    """List only immediate UUID directories; do not read reviews on startup."""
    import re
    parent = Path(directory)
    if parent.resolve() != parent.absolute():
        raise ValueError('Redirected UI run directory is unsupported')
    if not parent.exists():
        return []
    return sorted(p.name for p in parent.iterdir()
                  if re.fullmatch(r'[0-9a-f]{32}', p.name)
                  and p.is_dir() and p.resolve() == p.absolute())
