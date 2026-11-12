"""Evaluation-only locator diagnostics. Never imported by production batch modules."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def evaluate(records, key, *, ks=(1, 3, 5)):
    annotations = {r['id']: r for r in key['records']}
    ids = [r['id'] for r in records]
    if len(ids) != len(set(ids)) or not set(ids) <= annotations.keys():
        raise ValueError('Duplicate or unknown output IDs')
    hits = {str(k): 0 for k in ks}
    source_hits = {str(k): 0 for k in ks}
    eligible = 0
    failures = []
    for r in records:
        a = annotations[r['id']]
        expected = {(e['source'], e['page']) for e in a.get('evidence', [])}
        if not expected and a.get('expected_source') and a.get('expected_page'):
            expected = {(a['expected_source'], a['expected_page'])}
        if not expected:
            continue
        eligible += 1
        for k in ks:
            got = {(p['source'], p['page']) for p in r['retrieved_passages'][:k]}
            hits[str(k)] += bool(expected & got)
            source_hits[str(k)] += bool({s for s, _ in expected} & {s for s, _ in got})
        if not expected & {(p['source'], p['page']) for p in r['retrieved_passages'][:max(ks)]}:
            failures.append(dict(id=r['id'], expected=sorted(expected),
                retrieved=[(p['source'], p['page']) for p in r['retrieved_passages']],
                retrieval_error=r.get('retrieval_error')))
    return dict(evaluated_questions=len(records), dataset_questions=len(annotations),
        complete=set(ids) == set(annotations), locator_eligible_questions=eligible,
        source_page_hit_at_k={k:dict(hits=v, denominator=eligible, rate=v/eligible if eligible else None) for k,v in hits.items()},
        source_hit_at_k={k:dict(hits=v, denominator=eligible, rate=v/eligible if eligible else None) for k,v in source_hits.items()},
        support_classes=dict(Counter(r['evidence_assessment']['classification'] for r in records)),
        retrieval_errors=[r['id'] for r in records if r.get('retrieval_error')],
        empty_passages=[r['id'] for r in records if not r['retrieved_passages']],
        failures=failures, annotation_status=key.get('annotation_status'),
        human_verified=key.get('human_verified'),
        limitations=['Provisional synthetic model-authored answer key is not human gold.',
            'Hit means any annotated source/page matched, not all aspects covered or semantic correctness.',
            'Support classes are heuristic distributions, not measured classification accuracy.',
            'No answers, approved citations, operational effectiveness or model accuracy evaluated.'])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('retrieval_output', type=Path)
    parser.add_argument('answer_key', type=Path)
    parser.add_argument('report', type=Path)
    args=parser.parse_args(argv)
    records=json.loads(args.retrieval_output.read_text(encoding='utf-8'))
    key=json.loads(args.answer_key.read_text(encoding='utf-8'))
    report=evaluate(records,key)
    report['input_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.retrieval_output,args.answer_key)}
    with args.report.open('x',encoding='utf-8') as f:
        json.dump(report,f,indent=2)
        f.write('\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
