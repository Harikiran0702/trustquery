"""Finite policy-section paraphrases over twelve exposed extractions. No semantic QA."""
import hashlib
import io
import json
from pathlib import Path
from pypdf import PdfReader

BUNDLE = Path(__file__).with_name('review_policy_bundle.json')
BUNDLE_SHA256 = '8f7f6f90857217596c898c6d553a06f788935c73ad2328c924623ec107bca5fe'
POLICY_DIRECTORY = Path(__file__).resolve().parents[2] / 'data/sample'
SCOPE = 'experimental_proposed_policy_extraction_only'
WARNING = ('Experimental AI-assisted draft - synthetic/proposed/unapproved policy, not operating evidence. '
           'Not human approved. Machine-bound page references are unverified candidates, not human-verified citations. '
           'No adoption, implementation, compliance or independent safety validation is established.')


# Literal, case-sensitive full-question grammar. Slots are exact inspected titles;
# no normalization, keyword routing, suffix tolerance, or inferred requirements.
PARAPHRASE_TEMPLATES = (
    'Which requirements does the proposed policy specify for {title}?',
    'What does the proposed policy require for {title}?',
    'Please quote the complete proposed policy section on {title}.',
)


# Finite, inspected policy-only intent: no token scoring, normalization or suffix
# tolerance. Backup scope disambiguates encryption/incident-policy key passages.
# All forms return the existing COMPLETE section and mandatory document context.
BACKUP_KEY_TITLE = 'Encryption, keys and backup protection'
BACKUP_KEY_TEMPLATES = (
    'Quote the draft backup rules governing {subject}.',
    'What does the draft backup policy require for {subject}?',
)
BACKUP_KEY_SUBJECTS = (
    'old encryption key versions',
    'retaining older encryption key versions',
)


def matches_question(question, canonical):
    prefix = 'What requirements does the policy state for '
    if not canonical.startswith(prefix) or not canonical.endswith('?'):
        return False
    title = canonical[len(prefix):-1]
    if title == BACKUP_KEY_TITLE and any(
        question == template.format(subject=subject)
        for template in BACKUP_KEY_TEMPLATES for subject in BACKUP_KEY_SUBJECTS
    ):
        return True
    return question == canonical or any(
        question == template.format(title=title) for template in PARAPHRASE_TEMPLATES)


def eligible(record):
    return (record.get('evidence_assessment', {}).get('implementation_required') is False
            and record.get('outcome') not in ('processing_error', 'implementation_evidence_required'))


def proposal(record, *, enabled=False, _verify_sources=True):
    """Finite whole-question grammar; fresh full corpus/page verification on every call."""
    if enabled is not True or not eligible(record):
        raise ValueError('Abstained: explicit opt-in and policy-only scope required; implementation proof is unavailable')
    raw = BUNDLE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != BUNDLE_SHA256:
        raise ValueError('Abstained: experimental bundle version changed')
    cases = [c for c in json.loads(raw)['cases'] if matches_question(record['question'], c['question'])]
    if len(cases) != 1:
        raise ValueError('Abstained: no inspected section in the finite policy-question grammar; no generation attempted')
    case = cases[0]
    output = case['output']
    candidate = dict(bundle_sha256=BUNDLE_SHA256, case_id=case['case_id'], question=record['question'],
                     answer=output['answer'], references=output['validated_policy_citations'],
                     corpus_sha256=output['corpus_sha256'], human_approval=False, operational_proof=False)
    if not _verify_sources:
        return candidate
    pages = {}
    for source, expected in output['corpus_sha256'].items():
        if Path(source).name != source or not source.endswith('.pdf'):
            raise ValueError('Unsafe source')
        path = POLICY_DIRECTORY / source
        if path.resolve() != path.absolute() or path.stat().st_size > 20 * 1024 * 1024:
            raise ValueError('Unsafe or oversized source')
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('Abstained: source version/hash mismatch; fresh inspection required')
        pages[source] = [p.extract_text() or '' for p in PdfReader(io.BytesIO(data)).pages]
    for ref in output['validated_policy_citations']:
        page = ref['page']
        text = pages[ref['source']][page-1]
        digest = hashlib.sha256(json.dumps(text, sort_keys=True, ensure_ascii=True).encode()).hexdigest()
        if type(page) is not int or page < 1 or ref['quote'] != text or ref['page_sha256'] != digest:
            raise ValueError('Abstained: page binding mismatch')
    return candidate


def validate(record, metadata, answer=None, *, approve=False):
    expected = proposal(record, enabled=True, _verify_sources=approve)
    if metadata != expected:
        raise ValueError('Experimental provenance changed; reload the validated extraction')
    if approve and answer != expected['answer']:
        raise ValueError('Only the exact bound experimental extraction can be approved; edited claims need further validation')
    return expected
