"""Resumable demo cache; real API generation requires --gemini."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import time
from unittest.mock import patch
from trustquery.batch_retrieval import load_corpus, offline_mode
from trustquery.pipeline import ROOT, answer_question


class DailyQuotaReached(Exception):
    pass


class GeminiBusy(Exception):
    pass


def atomic_save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         delete=False, suffix='.tmp') as handle:
            name = handle.name
            json.dump(data, handle, indent=2, ensure_ascii=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def build_cache(questions, chunks, *, gemini=False, output=None, limit=15, ids=None):
    if limit <= 0:
        raise ValueError('limit must be positive')
    cache = json.loads(Path(output).read_text(encoding='utf-8')) if output and Path(output).exists() else {}
    rows = [r for r in questions if not ids or r.get('id') in ids]
    rows = [r for r in rows if not (gemini and
        (cache.get(r.get('id')) or cache.get(r['question']) or {}).get('method') == 'grounded_gemini')][:limit]
    calls, done = 0, 0
    backend_loaded, backend = False, None
    def get_backend():
        nonlocal backend_loaded, backend
        if not backend_loaded:
            backend_loaded = True
            try:
                from trustquery.evaluation.nli import load_nli_backend
                backend = load_nli_backend(local_files_only=True)
            except (ImportError, OSError, RuntimeError, ValueError):
                pass
        return backend
    def save():
        if output:
            atomic_save(output, cache)
    def attempt_text(prompt):
        nonlocal calls
        today = datetime.now(timezone.utc).date().isoformat()
        usage = cache.setdefault('__gemini_usage__', {})
        if usage.get('date') != today:
            usage.clear()
            usage.update(date=today, calls=0, last_call=0)
        if usage['calls'] >= 20 or usage.get('exhausted'):
            raise DailyQuotaReached()
        delay = max(0, 13 - (time.time() - usage['last_call']))
        if delay:
            time.sleep(delay)
        usage['calls'] += 1
        usage['last_call'] = time.time()
        calls += 1
        save()  # Count attempts durably before the network request, including failures.
        from trustquery.gemini_client import generate_text
        try:
            return generate_text(prompt, max_retries=1)
        except Exception as exc:
            status = getattr(exc, 'status_code', None) or getattr(exc, 'code', None)
            message = str(exc).lower()
            if str(status) == '429' or any(x in message for x in ('429', 'quota', 'resource_exhausted')):
                # Do not retry today; this also exceeds any Retry-After delay.
                usage['exhausted'] = True
                save()
                raise DailyQuotaReached() from exc
            if str(status) == '503' or any(x in message for x in ('503', 'high demand', 'unavailable')):
                raise GeminiBusy() from exc
            raise

    def live_text(prompt):
        for delay in (30, 60, 120):
            try:
                return attempt_text(prompt)
            except GeminiBusy:
                time.sleep(delay)
        return attempt_text(prompt)

    try:
        for row in rows:
            options = dict(chunks=chunks, offline=not gemini,
                question_id=row.get('id'), raise_generation_errors=True,
                nli_backend_factory=get_backend)
            if gemini:
                from trustquery.generation import answer_generator
                with patch.object(answer_generator, 'generate_text', live_text):
                    result = answer_question(row['question'], options)
            else:
                result = answer_question(row['question'], options)
            generated = any('grounded Gemini' in r for r in result['reasons'])
            cache[row.get('id', row['question'])] = dict(
                question=row['question'], method='grounded_gemini' if generated else 'offline_policy_extract',
                answer=result['answer'], citations=result['citations'])
            done += 1
            save()
    except GeminiBusy:
        save()
        print(f'Gemini busy: {done} of {len(rows)} done. Try again later.')
    except DailyQuotaReached:
        save()
        print(f'Daily quota reached: {done} of {len(rows)} done. Run again tomorrow.')
    finally:
        print(f'Total Gemini calls: {calls}')
    return cache


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--questions', type=Path, default=ROOT/'data/eval/clouddesk_questionnaire.json')
    parser.add_argument('--corpus', type=Path, default=ROOT/'data/sample')
    parser.add_argument('--output', type=Path, default=ROOT/'data/demo/answers_cache.json')
    parser.add_argument('--gemini', action='store_true')
    parser.add_argument('--limit', type=int, default=15)
    parser.add_argument('--ids', nargs='+', help='Question IDs, space or comma separated')
    args = parser.parse_args(argv)
    offline_mode()
    if args.gemini:
        from trustquery.config import get_gemini_api_key
        get_gemini_api_key()
    ids = {i for group in args.ids for i in group.split(',')} if args.ids else None
    build_cache(json.loads(args.questions.read_text(encoding='utf-8')),
        load_corpus(args.corpus, preserve_layout=True), gemini=args.gemini,
        output=args.output, limit=args.limit, ids=ids)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
