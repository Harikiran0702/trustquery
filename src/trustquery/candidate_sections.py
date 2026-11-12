"""Bounded offline section locators. Similarity never authorizes an answer."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re

from pypdf import PdfReader
from trustquery import review_policy
from trustquery.file_hashes import file_sha256, text_sha256
from trustquery.retrieval.bm25_retriever import search_bm25

MANIFEST = Path(__file__).with_name('candidate_sections_manifest.json')
# Fixed at installation, never derived from current source bytes at query time.
MANIFEST_SHA256 = 'a4d2f27075f8cb2623705dabc6f8d5a678ec0dc4b5167d4c9684e91ab5e2765b'
MIN_SCORE = 2.0
MIN_OVERLAP = 2
UNIQUE_MARGIN = 0.25
MAX_CANDIDATES = 3
STOP = set('a an the what which how who when where does do is are for of to in on and or policy proposed draft requirements require rules quote please under about'.split())
BLOCKED = re.compile(
    r'\b(actual|implemented|implementation|deployed|production|operating|certif\w*|compliance|compliant|sla|approved|approval status|guarantee\w*|prove|proof|confirm|ignore|except|without|excluding|not|never|contradict\w*)\b|\d', re.I)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def terms(text):
    return set(re.findall(r'[a-z]+', text.casefold())) - STOP


def _verified_blobs():
    """Freshly verify pinned source bytes without parsing PDF content."""
    raw = MANIFEST.read_bytes()
    if text_sha256(raw) != MANIFEST_SHA256:
        raise ValueError('Candidate manifest changed')
    manifest = json.loads(raw)
    folder = review_policy.POLICY_DIRECTORY
    if folder.resolve() != folder.absolute():
        raise ValueError('Redirected corpus')
    blobs = {}
    for name, expected in manifest['files'].items():
        if Path(name).name != name or Path(name).suffix not in ('.pdf', '.json'):
            raise ValueError('Unsafe source name')
        path = folder / name
        if path.resolve() != path.absolute() or path.stat().st_size > 20 * 1024 * 1024:
            raise ValueError('Unsafe or oversized source')
        data = path.read_bytes()
        if file_sha256(path, data) != expected:
            raise ValueError('Source hash mismatch: ' + name)
        blobs[name] = data
    return blobs, manifest['files']


def verified_source_hashes():
    """Current pinned PDF/JSON hashes for preflight drift checks only."""
    return _verified_blobs()[1]


def load_sections(*, include_legacy_layout=False):
    """Verify the entire eight-PDF/JSON corpus before exposing any locator.

    Only the existing extractor's one-numbered-section-per-page layout is
    supported. Access control's older layout is explicitly excluded, never
    silently interpreted as a complete section. All document pages are kept.
    """
    blobs, source_hashes = _verified_blobs()
    sections, excluded = [], []
    for name in sorted(n for n in blobs if n.endswith('.pdf')):
        annotation = json.loads(blobs[name[:-4] + '.json'])
        if (annotation.get('synthetic') is not True or
                annotation.get('approval_status') != 'not_recorded' or
                annotation.get('review_status') != 'requires_user_review'):
            raise ValueError('Unsupported source status')
        reader = PdfReader(io.BytesIO(blobs[name]))
        if reader.is_encrypted or not 1 <= len(reader.pages) <= 200:
            raise ValueError('Unsupported PDF')
        pages = [p.extract_text() or '' for p in reader.pages]
        headings = [list(re.finditer(r'(?m)^([1-9][0-9]*)\. ([^\n]+)$', t)) for t in pages]
        numbered = not any(len(h) != 1 or int(h[0][1]) != i for i, h in enumerate(headings, 1))
        if not numbered and not include_legacy_layout:
            excluded.append(name)
            continue
        if any(not t or len(t) > 100000 for t in pages):
            raise ValueError('Incomplete or oversized extracted context')
        refs = [dict(source=name, page=i, quote=t, pdf_sha256=source_hashes[name],
                     page_utf8_sha256=sha(t.encode()), independently_verified=False,
                     approval_status='not_recorded') for i, t in enumerate(pages, 1)]
        for i, (text, heading) in enumerate(zip(pages, headings), 1):
            sections.append(dict(text=text[heading[0].start():] if numbered else text,
                title=heading[0][2] if numbered else 'Full physical page (section layout unverified)',
                section_kind='numbered_section' if numbered else 'full_page',
                metadata=dict(source=name, page=i, chunk_index=i-1),
                document_context=refs))
    return sections, excluded, source_hashes


def locate(record, *, enabled=False):
    """Return read-only candidates and ALWAYS abstain. No store/editor payload.

    A unique lexical locator is not an intent match, entailment check, or
    contradiction resolution. No score can cross this draft boundary.
    """
    out = dict(method='bounded_bm25_full_section_candidates_v1', status='needs_evidence',
               answer='', citations=[], candidates=[], human_approval=False,
               operational_proof=False, source_approval='not_recorded',
               support_status='not_semantically_adjudicated', draft_eligible=False)
    if enabled is not True:
        return dict(out, reason='Explicit candidate opt-in required')
    question = record.get('question')
    if not isinstance(question, str) or not 1 <= len(question) <= 1000 or not question.strip():
        return dict(out, reason='Unsupported question size/type')
    if not review_policy.eligible(record) or BLOCKED.search(question):
        return dict(out, reason='Unsafe/operational/endorsement/value/exception scope; abstained')
    if not re.search(r'\b(policy|policies|draft|proposed)\b', question, re.I):
        return dict(out, reason='Explicit policy-text scope required')
    try:
        sections, excluded, hashes = load_sections()
        ranked = search_bm25(question, sections, top_k=len(sections))
        by_locator = {(s['metadata']['source'], s['metadata']['page']): s for s in sections}
        plausible = []
        for hit in ranked:
            section = by_locator[(hit['metadata']['source'], hit['metadata']['page'])]
            overlap = sorted(terms(question) & terms(section['text']))
            if hit['bm25_score'] >= MIN_SCORE and len(overlap) >= MIN_OVERLAP:
                plausible.append(dict(source=hit['metadata']['source'], page=hit['metadata']['page'],
                    title=section['title'], section_text=section['text'], score=hit['bm25_score'],
                    overlap=overlap, document_context=section['document_context'],
                    evidence_status='UNVERIFIED CANDIDATE'))
        out.update(candidates=plausible[:MAX_CANDIDATES], excluded_layouts=excluded,
                   corpus_sha256=hashes, plausible_section_count=len(plausible))
        unique = bool(plausible) and (len(plausible) == 1 or
            (plausible[0]['score'] - plausible[1]['score']) / plausible[0]['score'] >= UNIQUE_MARGIN)
        out['locator_status'] = 'unique_lexical_locator' if unique else 'ambiguous_or_missing_locator'
        out['reason'] = ('Candidates only: lexical uniqueness cannot establish intent, completeness, '
                         'negation/exception entailment or absence of contradictory evidence. Human review required.')
    except (ValueError, OSError, KeyError, TypeError) as exc:
        out.update(candidates=[], reason='Integrity/layout failure: ' + str(exc))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('question')
    parser.add_argument('--enable-section-candidates', action='store_true')
    args = parser.parse_args(argv)
    print(json.dumps(locate(dict(question=args.question, outcome='abstained',
        evidence_assessment={'implementation_required': False}),
        enabled=args.enable_section_candidates), indent=2))


if __name__ == '__main__':
    main()
