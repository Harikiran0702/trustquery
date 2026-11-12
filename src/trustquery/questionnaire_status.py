"""Reportable preflight dispositions, separate from saved human decisions."""
from collections import Counter
import csv
import io

LABELS = {'draft_eligible': 'Eligible for unverified draft',
          'needs_evidence': 'Needs evidence', 'needs_review': 'Needs review',
          'abstained': 'Other justified abstention'}

# Stable codes derived from allowlisted gate messages, including v1 saved runs.
REASONS = {
    'Policy-only intent unclear; implementation evidence required': 'policy_intent_unclear',
    'Operational/certification/audit proof cannot be established from proposed policies': 'operating_proof_required',
    'Untrusted instruction or requested omission': 'unsafe_request',
    'Known ambiguity or conflict; human resolution required': 'ambiguity_or_conflict',
    'Prior processing error': 'processing_error',
    'Potential cross-document conflict; explicit human resolution required': 'potential_conflict',
    'Insufficient candidate relevance; no draft request': 'insufficient_relevance',
    'Saved retrieval/source drift; create a fresh run': 'source_or_retrieval_drift',
    'Questionable source instructions': 'unsafe_source',
    'Full corpus exceeds bounded context; no truncation or call': 'context_limit',
}


def disposition(result):
    reasons = list(result.get('reasons', []))
    codes = [REASONS.get(r, 'unclassified_gate') for r in reasons]
    # Integrity and unsafe-input failures take priority over evidence requests.
    if any(c in codes for c in ('unsafe_request','unsafe_source','processing_error','context_limit','source_or_retrieval_drift')):
        status = 'abstained'
    elif 'operating_proof_required' in codes:
        status = 'needs_evidence'
    elif reasons or not result.get('eligible'):
        status = 'needs_review'
    else:
        status = 'draft_eligible'
    if not reasons and status != 'draft_eligible':
        reasons = ['Preflight unavailable; create a fresh offline run or inspect this question.']
        codes = ['preflight_unavailable']
    if status == 'draft_eligible':
        reasons = ['Explicit policy intent and candidate relevance permit an unverified request; support remains unproven.']
        codes = ['unverified_preflight_only']
    action = {
        'draft_eligible': 'Optionally request one draft, then review all claims and complete policy context.',
        'needs_evidence': 'Obtain relevant operating records, implementation proof or independent assurance; synthetic proposed policies cannot supply it.',
        'needs_review': 'Clarify policy-only scope and inspect candidate evidence, missing support and possible conflicts.',
        'abstained': 'Resolve the stated safety, integrity, processing or context issue before requesting a draft.',
    }[status]
    return dict(disposition=status, reason_codes=codes, disposition_reasons=reasons, next_action=action)


def report(rows, *, source_hashes=None, source_error=False):
    result=[]
    for r in rows:
        saved=r.get('general_rag', {})
        preflight=dict(saved, reasons=list(saved.get('reasons', [])))
        if source_error or (source_hashes is not None and saved and saved.get('corpus_sha256') != source_hashes):
            preflight['eligible']=False
            preflight['reasons'].append('Saved retrieval/source drift; create a fresh run')
        result.append(dict(id=r['id'], question=r['question'], **disposition(preflight)))
    return result


def counts(items):
    found=Counter(item['disposition'] for item in items)
    return {key: found[key] for key in LABELS}


def filter_ids(items, status='All'):
    return {item['id'] for item in items if status == 'All' or item['disposition'] == status}


def csv_bytes(items):
    stream=io.StringIO(newline='')
    writer=csv.writer(stream)
    writer.writerow(['question_id','question','preflight_disposition','reason_codes','reasons','next_action'])
    def literal(value):
        # CSV is also opened in spreadsheets: neutralize formula prefixes.
        return "'"+value if value.lstrip().startswith(('=','+','-','@')) else value
    for item in items:
        writer.writerow([literal(item['id']),literal(item['question']),item['disposition'],
            '; '.join(item['reason_codes']),'; '.join(item['disposition_reasons']),item['next_action']])
    return stream.getvalue().encode('utf-8-sig')
