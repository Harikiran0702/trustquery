"""Compare explicit bounded human judgments to frozen offline pilot outputs."""
import json
from collections import Counter
from pathlib import Path

from trustquery.file_hashes import legacy_crlf_sha256, text_sha256

LABELS = {'text_supported', 'contradicted', 'conflicted', 'insufficient_evidence',
          'implementation_evidence_required', 'partial'}
ROOT = Path(__file__).resolve().parents[3]

def compare(ledger, root=ROOT):
    if ledger['formal_approval'] is not False or ledger['generation_blocked'] is not True:
        raise ValueError('provenance gates must remain closed')
    required = {'data/eval/phase4_step3e/evidence_map.json',
                'data/eval/phase4_step3e/decisions.md',
                'data/eval/phase4_step3d/worksheet.json',
                'data/eval/phase4_step3c/results.json'}
    if set(ledger['source_sha256']) != required:
        raise ValueError('missing or unexpected source bindings')
    for path, digest in ledger['source_sha256'].items():
        data = (root/path).read_bytes()
        # Frozen ledgers recorded a mix of LF and CRLF digests.
        if digest not in (text_sha256(data), legacy_crlf_sha256(data)):
            raise ValueError('source drift: ' + path)
    worksheet = json.loads((root/'data/eval/phase4_step3d/worksheet.json').read_text())
    pilot = json.loads((root/'data/eval/phase4_step3c/results.json').read_text())
    cases = {c['id']: c for c in worksheet['cases']}
    predictions = {c['id']: c for c in pilot['results']}
    ids = [c['case_id'] for c in ledger['cases']]
    if len(ids) != len(set(ids)) or set(ids) != set(cases) or set(ids) != set(predictions):
        raise ValueError('case coverage mismatch')
    rows = []
    pending = []
    for record in ledger['cases']:
        cid = record['case_id']
        component_ids = [c['id'] for c in record['components']]
        if (len(component_ids) != len(set(component_ids)) or
            set(component_ids) != {c['id'] for c in cases[cid]['claims']}):
            raise ValueError('component coverage mismatch')
        if record['state'] not in {'pending', 'ambiguous', 'deferred', 'confirmed'}:
            raise ValueError('invalid review state')
        labels = [record['case_label']] + [c['label'] for c in record['components']]
        if any(label is not None and label not in LABELS for label in labels):
            raise ValueError('invalid label')
        if record['state'] != 'confirmed':
            pending.append(cid)
            continue
        if (record['case_label'] is None or (any(c['label'] is None for c in record['components']) and not record.get('component_review_pending'))) or not all(record.get(k) for k in
                ('source_message_id', 'reason', 'scope')):
            raise ValueError('incomplete confirmed judgment')
        # Reviewer names and timestamps remain null unless explicitly supplied.
        row = dict(case_id=cid, human=record['case_label'], components=record['components'])
        for method in ('baseline', 'nli'):
            pred = predictions[cid]['semantic' if method == 'nli' else method]
            pcomponents = {c['id']: c['status'] for c in pred['components']}
            if set(pcomponents) != set(component_ids):
                raise ValueError('prediction component mismatch')
            row[method] = dict(status=pred['status'], components=pcomponents)
        rows.append(row)
    metrics = {}
    for method in ('baseline', 'nli'):
        metrics[method] = {}
        for level in ('case', 'component'):
            pairs = []
            for row in rows:
                pairs.extend([(row['case_id'], row['human'], row[method]['status'])] if level == 'case'
                    else [(row['case_id'] + ':' + c['id'], c['label'], row[method]['components'][c['id']])
                          for c in row['components'] if c['label'] is not None])
            metrics[method][level] = dict(count=len(pairs),
                agreement=sum(h == p for _, h, p in pairs)/len(pairs) if pairs else None,
                confusion=dict(Counter(h + ' -> ' + p for _, h, p in pairs)),
                disagreements=[i for i,h,p in pairs if h != p],
                false_supported=[i for i,h,p in pairs if p == 'text_supported' and h != 'text_supported'])
    return dict(corrected_claim_decisions=ledger.get('corrected_claim_decisions', []), component_judgments_pending=[r['case_id'] for r in ledger['cases'] if r.get('component_review_pending')], confirmed_cases=len(rows), pending_cases=pending, metrics=metrics, rows=rows,
                formal_approval=False, generation_blocked=True,
                limitation='Bounded synthetic pilot agreement only; no deployment or operational proof.')

if __name__ == '__main__':
    folder = ROOT/'data/eval/phase4_step3f'
    ledger = json.loads((folder/'adjudications.json').read_text())
    (folder/'comparison.json').write_text(json.dumps(compare(ledger), indent=2) + '\n')

