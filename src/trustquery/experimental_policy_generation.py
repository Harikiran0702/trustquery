"""Finite, inspected policy rewrite plans; opt-in and never imported by runtime.

Plans are authored experimental annotations, not an entailment model or human
approval. Exact question matching is an intent allowlist, never a sentence-support
fallback. Every PDF in the corpus is byte-bound, including unselected context.
"""
import copy
import hashlib
import json
from pathlib import Path
from pypdf import PdfReader
from trustquery.experimental_generation import BOUNDARY, digest

PLAN_PATH = Path(__file__).with_name('experimental_policy_plans.json')


def norm(text):
    return ' '.join(text.split())


def generate_policy_draft(question, passages, corpus, *, enabled=False,
                          typed_claims=None, generator=None):
    """Corpus maps PDF basenames to bytes, NOT caller-declared hashes/status.

    Selected passages must be exact normalized spans of the bound current PDF.
    A scoped typed-claim request must match the entire authored component plan.
    No labels, answer keys, adjudication decisions or NLI predictions are read.
    """
    out = dict(experimental=True, method='version_bound_policy_plans_v1',
        status='needs_evidence', answer='', claims=[], citations=[],
        validated_policy_citations=[], operational_proof=False, human_approval=False,
        review_status='requires_human_review', reason='', source_approval='not_recorded')
    if enabled is not True:
        out['reason'] = 'Explicit experimental opt-in required'
        return out
    try:
        config = json.loads(PLAN_PATH.read_text(encoding='utf-8'))
        plans = [p for p in config['profiles'] if p['question'] == question]
        if len(plans) != 1:
            raise ValueError('Question has no inspected policy-only plan')
        plan = plans[0]
        if typed_claims is not None and typed_claims != plan['typed_claims']:
            raise ValueError('Typed claim scope is unsupported, mixed or ambiguous')
        if not isinstance(passages, list) or not passages:
            raise ValueError('Selected evidence is missing')
        if set(corpus) != set(config['corpus_sha256']):
            raise ValueError('Complete bound corpus required')
        pages = {}
        import io
        for name, expected_hash in config['corpus_sha256'].items():
            data = corpus[name]
            if not isinstance(data, bytes) or hashlib.sha256(data).hexdigest() != expected_hash:
                raise ValueError('Document/version mismatch; plan requires fresh inspection')
            pages[name] = [p.extract_text() or '' for p in PdfReader(io.BytesIO(data)).pages]
        anchors = []
        for p in passages:
            source, page, text = p['source'], p['page'], p['text']
            if (source not in pages or type(page) is not int or not 1 <= page <= len(pages[source])
                    or not isinstance(text, str) or not norm(text)
                    or norm(text) not in norm(pages[source][page-1])):
                raise ValueError('Selected passage does not match bound source/page')
            if source == plan['source'] and page == plan['anchor_page']:
                anchors.append(text)
        if not anchors:
            raise ValueError('Required evidence page was not selected')
        # Require the complete authoritative section in selected evidence. No
        # arbitrary sentence can enable a plan using unrelated text on that page.
        anchor_text = plan['anchor_section']
        if not any(anchor_text in norm(t) for t in anchors):
            raise ValueError('Selected evidence omits required section context')
        expected = []
        refs = []
        for c in plan['claims']:
            source, page = plan['source'], c['page']
            ref = dict(source=source, page=page, evidence_id=f'{source}:p{page}',
                quote=pages[source][page-1], pdf_sha256=config['corpus_sha256'][source],
                page_sha256=digest(pages[source][page-1]), approval_status='not_recorded',
                validation='byte_bound_inspected_policy_plan',
                role='policy_text_only_not_operational_or_human_approval')
            refs.append(ref)
            expected.append(dict(text='POLICY REQUIREMENT (synthetic, proposed): '+c['text'],
                                 evidence_id=ref['evidence_id']))
        proposal = generator(json.dumps(dict(claims=expected))) if generator else dict(claims=copy.deepcopy(expected))
        if isinstance(proposal, str):
            proposal = json.loads(proposal)
        if not isinstance(proposal, dict) or set(proposal) != {'claims'} or proposal['claims'] != expected:
            raise ValueError('Claim rewrite/citation plan changed')
        answer = '\n'.join(f"{c['text']} [Source: {r['source']}, Page: {r['page']}; approval_status: not_recorded]"
                           for c,r in zip(expected,refs))
        out.update(status='experimental_draft', answer=answer+'\n'+BOUNDARY,
            claims=expected, validated_policy_citations=refs,
            corpus_sha256=copy.deepcopy(config['corpus_sha256']),
            request_sha256=digest([question, passages, typed_claims]),
            coverage='bounded policy scope only; no original-question completeness or operating claim',
            reason='Inspected plan and full corpus versions matched; selected section validated')
    except Exception as exc:
        out['reason'] = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
    return out
