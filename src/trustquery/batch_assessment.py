"""Conservative policy triage; lexical coverage is never semantic entailment."""
import re

STOP = set('a an the your our does do is are be been have has how what which who when where why often quickly written policy policies require requires required call for meant to and or of on in at by with before after as well can you share please describe provide under according must should shall'.split())
POLICY = re.compile(r'\b(policy|policies|written|plan|proposed governance|governance document|documented requirement|proposed requirement|process require\w*|how quickly would)\b', re.I)
# Domain nouns (production, testing, record) alone do not request operating proof.
OPERATIONS = re.compile(r'\b(actual|actually|deployed|implemented|effectiveness|recent|latest|last|currently|today|enforced|operating|completed|staffed|validated contacts|ready for use|is .{0,50}(?:enabled|in use|in place)|do you (?:use|enforce|maintain|operate)|have .{0,60}(?:tested|exercised|reassessed)|has .{0,60}been (?:tested|approved)|(?:provide|share|show) .{0,45}(?:evidence|logs?|records?|results?|examples?|inventory))\b', re.I)
CAUTION = re.compile(r'\b(not|never|no|except|unless|if|only|may|optional|uncertain|unsupported|tbd|unknown|unresolved|pending|contradict\w*)\b', re.I)
REQUIREMENT = re.compile(r'\b(must|shall|require\w*|reviews?|every|prohibit\w*|only|use|keep|store|record|test|assess|approve\w*|support|avoid|revoke|encrypt|allow)\b', re.I)


def terms(text):
    return sorted(set(re.findall(r'[a-z0-9]+', text.lower())) - STOP)


def assess(question, passages):
    """Expose sentence coverage and separate intent; never infer live controls.

    Full support is limited to a literal single-requirement template, with no
    modifiers, scope or extra clauses in its evidence sentence. All other
    lexical matches remain partial and require semantic review. No synonym
    expansion or answer-key input is used.
    """
    wanted = set(terms(question))
    policy = bool(POLICY.search(question))
    operational = bool(OPERATIONS.search(question)) or not policy
    details, covered, literal, bounded_partial = [], set(), False, False
    relevant_caution = False
    # A deliberately bounded parser, not bag-of-words completeness.
    query = re.fullmatch(r'Does (?:your )?(?:written )?policy require ([\w -]+)\?', question, re.I)
    target = query.group(1).strip() if query else None
    if target and re.search(r'\b(and|or|all|every|any|not|no|without|unless|except)\b', target, re.I):
        target = None
    status = []
    for p in passages:
        text = p['text']
        eligible = (p.get('evidence_type') == 'synthetic_policy_requirement'
                    and p.get('operational_proof') is False)
        if eligible and re.search(r'Approval_status:\s*not_recorded\b', text, re.I):
            status.append(dict(passage_id=p['chunk_id'], text='Approval_status: not_recorded',
                               status='source_approval_not_recorded'))
        sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
        unscoped_uncertainty = any(re.fullmatch(r'(?:TBD|Unknown|Pending)[.!]?', s.strip(), re.I) for s in sentences)
        matched, usable_sentences, blocked = set(), [], []
        for sentence in sentences:
            overlap = wanted & set(terms(sentence))
            cautions = sorted(set(m.group().lower() for m in CAUTION.finditer(sentence)))
            if eligible and cautions and (overlap or re.fullmatch(r'(?:TBD|Unknown|Pending)[.!]?', sentence.strip(), re.I)):
                blocked.append(dict(text=sentence, caution_terms=cautions))
                relevant_caution = True
            if eligible and overlap and REQUIREMENT.search(sentence) and not cautions and not unscoped_uncertainty:
                matched.update(overlap)
                usable_sentences.append(sentence)
                if target and re.fullmatch(r'Policy must require ' + re.escape(target) + r'\.', sentence, re.I):
                    literal = True
                if query:
                    components = re.split(r'\s+and\s+', query.group(1), flags=re.I)
                    if any(re.fullmatch(r'Policy must require ' + re.escape(c.strip()) + r'\.', sentence, re.I) for c in components):
                        bounded_partial = True
            if eligible and re.search(r'\b(?:this|proposed) policy\b.{0,120}\b(?:has not been approved|not approved)\b', sentence, re.I):
                status.append(dict(passage_id=p['chunk_id'], text=sentence,
                                   status='source_declares_not_approved'))
        covered.update(matched)
        details.append(dict(passage_id=p['chunk_id'], matched_terms=sorted(matched),
            caution_terms=sorted(set(m.group().lower() for m in CAUTION.finditer(text))),
            eligible_policy_requirement=bool(usable_sentences),
            usable_sentences=usable_sentences, blocked_sentences=blocked,
            classification='partially_supported' if matched else 'insufficient_evidence',
            reasons=['Sentence-level lexical coverage only; scope, polarity and completeness require review.'],
            operational_proof=False))
    complete = literal and not relevant_caution
    if operational:
        label = 'implementation_evidence_required'
        reason = 'Actual operation requested or policy-only intent unclear; synthetic text cannot prove operation.'
    elif complete:
        label = 'policy_supported'
        reason = 'Exact bounded requirement sentence; no operational claim or semantic paraphrase inferred.'
    elif bounded_partial:
        label = 'partially_supported'
        reason = 'A bounded requirement component is present; completeness or conflicting context remains unresolved.'
    else:
        label = 'insufficient_evidence'
        reason = 'No bounded support established; lexical coverage is exposed separately and cannot establish scope or semantic support.'
    for d in details:
        if operational:
            d['classification'] = label
        elif complete and d['matched_terms']:
            d['classification'] = label
        elif not bounded_partial:
            d['classification'] = 'insufficient_evidence'
    return dict(classification=label, method='bounded_policy_triage_v2',
        reasons=[reason] + (['Relevant caution-bearing sentences require polarity, exception or contradiction review.'] if relevant_caution else []),
        evidence_passage_ids=[d['passage_id'] for d in details if d['matched_terms']],
        matched_terms=sorted(covered), missing_terms=sorted(wanted-covered), passages=details,
        policy_requested=policy, implementation_required=operational,
        both_policy_and_implementation_required=policy and operational,
        policy_coverage='exact_bounded_requirement' if complete else 'partial_lexical' if covered else 'none',
        source_status_observations=status,
        uncertainty='Uncalibrated heuristic; no semantic entailment, contradiction resolution, adoption or effectiveness established. Human review required.',
        operational_proof=False, review_status='requires_human_review')
