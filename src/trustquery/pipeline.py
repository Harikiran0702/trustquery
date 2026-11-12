"""Grounded drafts for human review. Corpus is chunks, or chunks plus local backends."""
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import re

from trustquery.retrieval.bm25_retriever import query_terms as _terms


ROOT = Path(__file__).resolve().parents[2]
DEMO_CACHE = ROOT / 'data/demo/answers_cache.json'


def _sparse(question, chunks, top_k):
    from trustquery.retrieval.bm25_retriever import search_bm25
    return search_bm25(question, chunks, top_k, normalize_query=True)


def _operational(question):
    # Reuse intent classification only; the former assessment is not an answer gate.
    from trustquery.general_rag import POLICY, operating_proof
    return operating_proof(question) or not bool(POLICY.search(question))


def _hybrid(question, chunks, options):
    """Use the shared local-only ranking implementation."""
    from trustquery.general_rag import rank
    ranked, method, fallback = rank(question, chunks, hybrid=True)
    if fallback:
        raise OSError(fallback)
    return ranked[:options.get('top_k', 5)]


def _retrieve(question, chunks, options, reasons):
    if options.get('retrieve') is not None:
        results = options['retrieve'](query=question, chunks=chunks,
            collection=options.get('collection'), reranker_model=options.get('reranker_model'),
            top_k=options.get('top_k', 5), candidate_k=options.get('candidate_k', 10))
        reasons.append('Retrieval: supplied backend.')
    elif options.get('retrieval') == 'bm25':
        results = _sparse(question, chunks, options.get('top_k', 5))
        reasons.append('Retrieval: BM25 offline.')
    else:
        try:
            results = _hybrid(question, chunks, options)
            reasons.append('Retrieval: hybrid with local models.')
        except (ImportError, OSError, RuntimeError, ValueError):
            results = _sparse(question, chunks, options.get('top_k', 5))
            reasons.append('Retrieval: BM25 fallback; local hybrid models unavailable.')
    # Bind every result to current corpus; lexical overlap is relevance, not entailment.
    known = {(c['text'], json.dumps(c.get('metadata', {}), sort_keys=True)) for c in chunks}
    selected = []
    for c in results:
        if (c['text'], json.dumps(c.get('metadata', {}), sort_keys=True)) not in known:
            raise ValueError('Retrieved passage is not in the supplied corpus')
        metadata = c.get('metadata', {})
        if metadata.get('source') and type(metadata.get('page')) is int and metadata['page'] > 0 and _terms(question) & _terms(c['text']):
            selected.append(c)
    return selected[:options.get('top_k', 5)]


HEADER = re.compile(r'SYNTHETIC\s*/\s*DEMO\s*\|[\s\S]*?Page\s+\d+\s+of\s+\d+', re.I)


def _clean(text):
    return HEADER.sub('', text).strip()


def _sentences(text):
    text = _clean(text)
    lines = text.splitlines()
    text = '\n'.join(line for i, line in enumerate(lines) if not (
        line.strip() and not re.search(r'[.!?]', line) and i + 1 < len(lines)
        and re.match(r'[A-Z]', lines[i + 1].strip())))
    return [' '.join(m[1].split()) for m in re.finditer(
        r'(?:^|(?<=[.!?])\s+)([A-Z](?:(?![.!?](?:\s|$))[\s\S])*[.!?](?=\s|$))', text)
        if not any(tag in m[1].lower() for tag in ('synthetic', 'review_status:', 'approval_status:'))]


def _extract_bound(passages, question):
    wanted = _terms(question)
    vocabulary, frequencies = _corpus_terms(tuple(p['text'] for p in passages))
    weights = {t: math.log(1 + len(passages) / (1 + frequencies[t]))
               for t in wanted & vocabulary}
    candidates = []
    for index, passage in enumerate(passages):
        for sentence in _sentences(passage['text']):
            overlap = len(wanted & _terms(sentence))
            if overlap >= 2:
                score = sum(weights[t] for t in wanted & _terms(sentence))
                # Retrieval order breaks ties and gives the leading passage a modest preference.
                candidates.append((-score / (1 + .1 * index), index, sentence, passage))
    candidates.sort(key=lambda item: item[:3])
    chosen, refs, seen = [], [], set()
    for _, _, sentence, passage in candidates:
        if sentence in chosen:
            continue
        m = passage['metadata']
        locator = m['source'], m['page']
        chosen.append(sentence)
        if locator not in seen:
            refs.append(dict(source=locator[0], page=locator[1], quote=sentence))
            seen.add(locator)
        if len(chosen) == 3:
            break
    return ('The policy text states: ' + ' '.join(chosen) if chosen else ''), refs


def _extract(passages, question=''):
    return _extract_bound(passages, question)[0]


def _generation_issue(answer):
    if not isinstance(answer, str) or not answer.strip():
        return 'empty Gemini response'
    if re.match(r'^\s*insufficient\s+evidence\b', answer, re.I):
        return 'Gemini reported insufficient evidence'
    return None


def _cached(entry, chunks):
    if not isinstance(entry, dict) or entry.get('method') != 'grounded_gemini':
        return None
    if _generation_issue(entry.get('answer')):
        return None
    refs = entry.get('citations')
    if not isinstance(refs, list) or not 1 <= len(refs) <= 3:
        return None
    passages = []
    for ref in refs:
        if not isinstance(ref, dict) or not ref.get('source') or type(ref.get('page')) is not int or ref['page'] < 1:
            return None
        # Full evidence text records what generation received; a page locator or
        # extracted sentence alone cannot recover the original prompt evidence.
        matches = [c for c in chunks if (c.get('metadata', {}).get('source'),
            c.get('metadata', {}).get('page'), c['text']) ==
            (ref.get('source'), ref.get('page'), ref.get('quote'))]
        if not matches:
            return None
        passages.append(matches[0])
    if len({(c['metadata']['source'], c['metadata']['page']) for c in passages}) != len(passages):
        return None
    return entry['answer'], passages


def _generation_passages(passages):
    selected, seen = [], set()
    for passage in passages:
        m = passage['metadata']
        locator = m['source'], m['page']
        if locator not in seen:
            selected.append(passage)
            seen.add(locator)
        if len(selected) == 3:
            break
    return selected


def _generate(question, passages):
    from trustquery.generation.answer_generator import build_grounded_prompt, generate_text
    return generate_text(build_grounded_prompt(question, passages))


def _faithfulness(answer, passages, options):
    from trustquery.evaluation.faithfulness import answer_faithfulness
    from trustquery.evaluation.nli import load_nli_backend
    backend = options['nli_backend_factory']() if 'nli_backend_factory' in options else (options['nli_backend'] if 'nli_backend' in options else load_nli_backend(local_files_only=True))
    if backend is None:
        raise OSError('NLI unavailable')
    return answer_faithfulness(answer, passages, backend).score


@lru_cache(maxsize=8)
def _corpus_terms(texts):
    terms = [_terms(_clean(text)) for text in texts]
    vocabulary = set().union(*terms)
    return vocabulary, {t: sum(t in row for row in terms) for t in vocabulary}


def _relevant(question, chunks, passages):
    """Use missing technical identifiers and IDF-weighted passage support."""
    wanted = _terms(question)
    vocabulary, frequencies = _corpus_terms(tuple(c['text'] for c in chunks))
    identifiers = re.findall(r'\b[A-Z][A-Z0-9]{1,}\b|\b[a-z]+(?:-[a-z]+)+\b', question)
    if any(not _terms(t) <= vocabulary for t in identifiers):
        return False
    weights = {t: math.log(1 + len(chunks) / (1 + frequencies[t])) for t in wanted & vocabulary}
    if not weights:
        return False
    # A numeric commitment must co-occur with the question's most distinctive
    # subject, rather than borrowing an unrelated number elsewhere on the page.
    numbers = set(re.findall(r'\b\d+\b', question))
    if numbers:
        distinctive = max(sorted(t for t in weights if not t.isdigit()), key=weights.get, default=None)
        if not any(numbers <= _terms(sentence) and distinctive in _terms(sentence)
                   for c in passages for sentence in _sentences(c['text'])):
            return False
    support = max(sum(w for t, w in weights.items() if t in _terms(c['text'])) / sum(weights.values()) for c in passages)
    if support < .25:
        return False
    # Broad evidence-gap rule: abstain when the strongest matching sentence
    # explicitly says the capability/approval is absent or undecided.
    sentences = [s for c in passages for s in _sentences(c['text'])]
    def match(sentence):
        return sum(w for t, w in weights.items() if t in _terms(sentence))
    best = max((match(s) for s in sentences), default=0)
    for sentence in sentences:
        if re.search(r'\bTBD\b|not assumed capabilities|has not been approved|approval not recorded|not (?:supported|available|offered|enabled)|does not (?:support|offer)', sentence, re.I):
            if len(wanted & _terms(sentence)) >= 2 and match(sentence) >= best:
                return False
    return True


def _coverage(wanted, passages):
    matched = max((len(wanted & _terms(_clean(p['text']))) for p in passages), default=0)
    return matched / max(1, len(wanted))


def answer_question(question, corpus, *, question_id=None):
    """Return an unapproved draft; dictionary corpus can supply retrieval/NLI backends.

    API generation requires an environment key and no explicit offline override.
    Cache references must bind to the current corpus; all drafts need review.
    """
    options = corpus if isinstance(corpus, dict) else {'chunks': corpus}
    chunks = options['chunks']
    operational = _operational(question)
    result = dict(answer='', citations=[], confidence='none',
                  needs_implementation_evidence=operational, reasons=[])
    reasons = result['reasons']
    if not question.strip() or not chunks:
        reasons.append('No relevant policy evidence found')
        return result
    try:
        passages = _retrieve(question, chunks, options, reasons)
    except Exception as exc:
        reasons.append('Retrieval failed: ' + type(exc).__name__)
        return result
    relevant = bool(passages) and _relevant(question, chunks, passages)
    answer = ''
    generated = False
    generation_passages = _generation_passages(passages)
    passages = generation_passages
    if relevant and os.environ.get('GEMINI_API_KEY') and not options.get('offline'):
        try:
            instruction = ('Answer only what the policy text says. Do not assert actual implementation, audits or certification. ' + question)
            answer = _generate(instruction, generation_passages)
            issue = _generation_issue(answer)
            generated = issue is None
            reasons.append('Generation: grounded Gemini draft; unverified source references.' if generated
                else f'Generation unavailable: {issue}.')
        except Exception as exc:
            if options.get('raise_generation_errors'):
                raise
            reasons.append('Generation unavailable: ' + type(exc).__name__)
    cache_chunks = options.get('demo_cache_chunks')
    if not generated and (not options.get('offline') or cache_chunks is not None):
        try:
            cache = json.loads(Path(options.get('cache_path', DEMO_CACHE)).read_text(encoding='utf-8'))
            if not isinstance(cache, dict):
                raise ValueError('Invalid cache schema')
            entry = cache.get(question_id or options.get('question_id')) or cache.get(question)
            if entry is None:
                entry = next((e for e in cache.values() if isinstance(e, dict) and e.get('question') == question), None)
        except (OSError, ValueError):
            entry = None
        if isinstance(entry, dict) and entry.get('question', question) != question:
            entry = None
        cached = _cached(entry, cache_chunks if cache_chunks is not None else chunks)
        if cached:
            answer, passages = cached
            generated = True
            reasons.append('Generation: cached draft; references remain unverified.')
        elif isinstance(entry, dict) and entry.get('method') == 'grounded_gemini':
            issue = _generation_issue(entry.get('answer')) or 'provenance not bound to corpus'
            reasons.append(f'Cached Gemini unavailable: {issue}.')
        else:
            reasons.append('Demo cache missing or not bound to the question.')
    if generated:
        citations = [dict(source=p['metadata']['source'], page=p['metadata']['page'],
                          quote=p['text']) for p in passages]
    elif relevant:
        answer, citations = _extract_bound(passages, question)
        why = 'offline mode' if options.get('offline') else 'no usable Gemini draft (see generation reasons)'
        reasons.append(f'Using policy extract: {why}.')
    else:
        answer = ''
    if not answer:
        reasons.append('No relevant policy evidence found')
        return result
    prefix = 'Policy-based draft (implementation evidence still required):'
    if operational:
        answer = answer.strip()
        while answer.startswith(prefix):
            answer = answer[len(prefix):].lstrip()
        reasons.append('Policy evidence does not establish implementation, audit or certification.')
    wanted = _terms(question)
    coverage = _coverage(wanted, generation_passages)
    if passages is not generation_passages:
        # A cached draft may have been generated with a different ranking.
        coverage = min(coverage, _coverage(wanted, passages))
    reasons.append(f'Retrieval term coverage: {coverage:.2f}.')
    confidence = 'low' if operational or coverage < .5 else 'medium'
    if options.get('offline'):
        reasons.append('Faithfulness check skipped in offline mode; uncertainty remains.')
    else:
        try:
            score = _faithfulness(answer, passages, options)
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError('Invalid faithfulness score')
            reasons.append(f'Faithfulness score: {score:.2f}; model estimate, not human verification.')
            confidence = 'high' if score >= .8 else 'medium' if score >= .5 else 'low'
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            reasons.append('Faithfulness check skipped: ' + type(exc).__name__ + '; uncertainty remains.')
    if operational:
        answer = prefix + '\n\n' + answer
        confidence = 'low'
    result.update(answer=answer, citations=citations, confidence=confidence)
    return result
