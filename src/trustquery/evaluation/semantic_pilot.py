"""Offline experiment: typed claims and selected evidence, never answer generation.

Claim decomposition/type and evidence selection are supplied review tasks, not
solved by NLI. Synthetic policy text cannot establish operational facts.
"""
import re
from dataclasses import asdict
from trustquery.evaluation.nli import verify_claim

TYPES = {'policy_requirement', 'operational_fact', 'source_status', 'unknown'}

def verify_components(claims, passages, backend=None):
    ids = [c['id'] for c in claims]
    pids = [p['chunk_id'] for p in passages]
    if len(ids) != len(set(ids)) or len(pids) != len(set(pids)):
        raise ValueError('duplicate claim or passage IDs')
    results = []
    for claim in claims:
        kind = claim['type']
        if kind not in TYPES or not claim['text'].strip():
            raise ValueError('invalid typed claim')
        pairs = []
        eligible = [p for p in passages if p.get('evidence_type') == 'synthetic_policy_requirement'
                    and p.get('operational_proof') is False and p.get('source')
                    and isinstance(p.get('page'), int) and p['page'] > 0 and p['text'].strip()]
        if kind == 'operational_fact':
            status = 'implementation_evidence_required'
        elif kind == 'unknown' or not eligible:
            status = 'insufficient_evidence'
        else:
            for p in eligible:
                entry = dict(passage_id=p['chunk_id'], source=p['source'], page=p['page'])
                if backend is None:
                    # Only exact complete sentence identity; lexical overlap never suffices.
                    sentences = re.split(r'(?<=[.!?])\s+|\n+', p['text'])
                    entry['decision'] = 'entailment' if claim['text'] in sentences else 'inconclusive'
                else:
                    try:
                        nli = verify_claim(claim['text'], p['text'], backend)
                        entry['scores'] = asdict(nli)
                        entry['decision'] = (nli.label if max(nli.entailment, nli.contradiction) >= .90
                                             and nli.neutral < .10 else 'inconclusive')
                    except (ValueError, RuntimeError) as error:
                        entry.update(decision='inconclusive', error=str(error))
                pairs.append(entry)
            decisions = {p['decision'] for p in pairs}
            if 'contradiction' in decisions:
                status = 'conflicted' if 'entailment' in decisions else 'contradicted'
            elif 'entailment' in decisions:
                status = 'text_supported'
            else:
                status = 'insufficient_evidence'
        results.append(dict(id=claim['id'], type=kind, status=status, pairs=pairs))
    statuses = {r['status'] for r in results}
    overall = ('implementation_evidence_required' if 'implementation_evidence_required' in statuses
               else 'conflicted' if 'conflicted' in statuses
               else 'contradicted' if 'contradicted' in statuses
               else 'text_supported' if results and statuses == {'text_supported'}
               else 'partial' if 'text_supported' in statuses else 'insufficient_evidence')
    return dict(status=overall, components=results, operational_proof=False,
                review_status='requires_human_signoff', source_approval='not_established',
                answers=[], citations=[])
