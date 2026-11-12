"""Offline calibration on the provisional 100-record questionnaire answer key."""
import argparse
import json
from pathlib import Path
from trustquery.batch_retrieval import load_corpus, offline_mode
from trustquery.pipeline import ROOT, _hybrid, _sparse, answer_question


def candidates_for(question, chunks, retrieval):
    if retrieval == 'hybrid':
        return _hybrid(question, chunks, {'top_k': len(chunks)})
    return _sparse(question, chunks, max(1, len(chunks)))


def evaluate(rows, chunks, retrieval='bm25'):
    ranks, documents, outcomes, citation_count = [], [], [], 0
    classes = {}
    for row in rows:
        candidates = candidates_for(row['question'], chunks, retrieval)
        result = answer_question(row['question'], dict(chunks=chunks, offline=True,
            retrieve=lambda **kwargs: candidates, top_k=5))
        answered = bool(result['answer'])
        outcomes.append((row['classification'] in ('policy-supported', 'partial', 'policy-only-not-implemented'), answered))
        classes.setdefault(row['classification'], dict(count=0, drafted=0, abstained=0))
        counts = classes[row['classification']]
        counts['count'] += 1
        counts['drafted'] += answered
        counts['abstained'] += not answered
        citation_count += len(result['citations']) if answered else 0
        if row['classification'] in ('policy-supported', 'partial', 'policy-only-not-implemented') and row.get('expected_source') and row.get('expected_page'):
            target = row['expected_source'], row['expected_page']
            ranks.append(next((i for i, c in enumerate(candidates, 1)
                if (c['metadata']['source'], c['metadata']['page']) == target), 0))
            documents.append(any(c['metadata']['source'] == target[0] for c in candidates[:5]))
    positive = sum(y for y, a in outcomes)
    negative = len(rows) - positive
    tp = sum(not y and not a for y, a in outcomes)
    abstained = sum(not a for y, a in outcomes)
    answered = len(rows) - abstained
    covered = sum(y and a for y, a in outcomes)
    return dict(classifications=classes, records=len(rows), answerable=positive, nonanswerable=negative,
        retrieval_denominator=len(ranks),
        page_hit1=sum(r == 1 for r in ranks)/len(ranks),
        page_hit3=sum(0 < r <= 3 for r in ranks)/len(ranks),
        MRR=sum(1/r for r in ranks if r)/len(ranks),
        document_hit5=sum(documents)/len(ranks),
        abstention_precision=tp/abstained if abstained else 0,
        abstention_recall=tp/negative, answerable_answered=covered,
        nonanswerable_abstained=tp, answered=answered,
        decision_accuracy=(covered+tp)/len(rows),
        avg_citations_per_nonempty_answer=citation_count/answered if answered else 0)


def load_rows(key_path, questions_path):
    key = json.loads(key_path.read_text(encoding='utf-8'))['records']
    questions = {r['id']: r['question'] for r in json.loads(questions_path.read_text(encoding='utf-8'))}
    assert len(key) == len({r['id'] for r in key}) == 100
    rows = [dict(id=r['id'], question=questions[r['id']], should_answer=r['should_answer'],
        expected_source=r['expected_source'], expected_page=r['expected_page'],
        classification=r['classification']) for r in key]
    assert len(rows) == 100
    return rows


def evaluate_unseen(path, chunks):
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return dict(status='unavailable_or_invalid_json')
    return _unseen_totals(payload, chunks, 'bm25')


def _unseen_totals(payload, chunks, retrieval):
    """Aggregate only; unknown labels are counted, never inferred."""
    rows = payload if isinstance(payload, list) else (payload.get('records', payload.get('questions'))
        if isinstance(payload, dict) else None)
    if not isinstance(rows, list):
        return dict(status='unrecognized_schema')
    totals = dict(records=len(rows), evaluated=0, unrecognized=0,
                  should_answer=0, should_abstain=0, drafted=0, abstained=0,
                  answerable_drafted=0, nonanswerable_abstained=0)
    for row in rows:
        if not isinstance(row, dict):
            totals['unrecognized'] += 1
            continue
        metadata = row.get('metadata')
        expected = metadata.get('should_answer') if isinstance(metadata, dict) else None
        question = row.get('question')
        if type(expected) is not bool or not isinstance(question, str) or not question.strip():
            totals['unrecognized'] += 1
            continue
        candidates = candidates_for(question, chunks, retrieval)
        result = answer_question(question, dict(chunks=chunks, offline=True,
            retrieve=lambda **kwargs: candidates, top_k=5))
        drafted = bool(result['answer'])
        totals['evaluated'] += 1
        totals['should_answer'] += expected
        totals['should_abstain'] += not expected
        totals['drafted'] += drafted
        totals['abstained'] += not drafted
        totals['answerable_drafted'] += expected and drafted
        totals['nonanswerable_abstained'] += not expected and not drafted
    return totals


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT/'data/eval/clouddesk_questionnaire_answer_key.json')
    parser.add_argument('--questions', type=Path, default=ROOT/'data/eval/clouddesk_questionnaire.json')
    parser.add_argument('--corpus', type=Path, default=ROOT/'data/sample')
    parser.add_argument('--include-holdout', action='store_true',
        help='Aggregate unseen report only after pipeline and thresholds are frozen')
    parser.add_argument('--hybrid', action='store_true', help='Compare with pinned local hybrid models')
    args = parser.parse_args(argv)
    offline_mode()
    rows = load_rows(args.dataset, args.questions)
    chunks = load_corpus(args.corpus, preserve_layout=True)
    print('Offline calibration on provisional labels; candidate ranks before answer filtering.')
    modes = ['bm25', 'hybrid'] if args.hybrid else ['bm25']
    results = {}
    for mode in modes:
        try:
            results[mode] = evaluate(rows, chunks, mode)
        except OSError as exc:
            if mode != 'hybrid':
                raise
            print(f'Hybrid unavailable: {exc}')
            results[mode] = None

    def columns(labels, value):
        print(f"{'Metric':34}" + ''.join(f'{m.upper():>22}' for m in modes))
        for label in labels:
            print(f'{label:34}' + ''.join(
                f'{value(m, label):>22}' if results[m] is not None else f'{"unavailable":>22}'
                for m in modes))

    classes = results['bm25']['classifications']
    print('Classifications (count / drafted / abstained):')
    columns(classes, lambda m, name: ' / '.join(str(results[m]['classifications'][name][k])
            for k in ('count', 'drafted', 'abstained')))
    metrics = [k for k in results['bm25'] if k != 'classifications']
    columns(metrics, lambda m, k: f'{results[m][k]:.6g}')
    if args.include_holdout:
        print('Unseen questions (aggregate only):')
        try:
            payload = json.loads((ROOT/'data/eval/holdout_questions.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            payload = None
        unseen = {}
        for mode in modes:
            if results[mode] is not None:
                try:
                    unseen[mode] = _unseen_totals(payload, chunks, mode)
                except OSError as exc:
                    print(f'{mode} unavailable on unseen questions: {exc}')
                    results[mode] = None
        columns(unseen.get('bm25', {}), lambda m, k: unseen[m][k])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
