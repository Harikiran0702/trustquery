"""Pipeline drafts with source references and a separate human review workflow."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import re

from trustquery.batch_retrieval import (
    load_corpus, offline_mode, open_index, retrieve_drafts as retrieve_drafts, validate_input,
)


def finalize(record):
    """Create a bounded extract or abstention without promoting lexical triage."""
    result = copy.deepcopy(record)
    passages = result['retrieved_passages']
    assessment = result['evidence_assessment']
    result.update(answer='', citations=[], review_status='requires_human_review',
                  operational_proof=False, errors=[])
    # Candidate references are grounded retrieval locators, never answer support.
    result['evidence_references'] = [dict(
        chunk_id=p['chunk_id'], source=p['source'], page=p['page'],
        chunk_index=p['chunk_index'], quote=p['text'], role='candidate_for_review',
        operational_proof=False) for p in passages]
    if 'retrieval_error' in result:
        result['errors'].append(dict(stage='retrieval', **result['retrieval_error']))
        outcome = 'processing_error'
        message = 'Retrieval failed. Review the recorded error and retry this question.'
    elif assessment['implementation_required']:
        outcome = 'implementation_evidence_required'
        message = 'Policy text cannot establish actual operation. Obtain implementation evidence and human review.'
    else:
        outcome = 'abstained'
        message = 'No safely bounded answer established. Review candidate passages for scope, exceptions and completeness.'
        match = re.fullmatch(r'Does (?:your )?(?:written )?policy require ([\w -]+)\?', result['question'], re.I)
        # Even a policy_supported heuristic is not sufficient: the entire retrieved
        # context must be a single literal requirement, with no hidden qualifiers.
        if match and assessment['classification'] == 'policy_supported':
            sentence = 'Policy must require ' + match.group(1).strip() + '.'
            if passages and all(p['text'].strip().casefold() == sentence.casefold() for p in passages):
                p = passages[0]
                quote = p['text'].strip()
                result['answer'] = ('The retrieved synthetic policy text states: "' + quote +
                    '" Policy adoption and implemented controls are not established.')
                result['citations'] = [dict(chunk_id=p['chunk_id'], source=p['source'],
                    page=p['page'], chunk_index=p['chunk_index'], quote=quote,
                    role='literal_policy_text_only', operational_proof=False)]
                outcome = 'policy_extract_draft'
                message = 'Literal extract only; human review required before use.'
    result['outcome'] = outcome
    result['review_note'] = message
    result['status'] = 'pending' if outcome == 'policy_extract_draft' else 'needs_evidence'
    result['provenance'].update(result_method='bounded_extract_or_abstain_v1',
        generation_method='none', experimental_nli_enabled=False)
    return result


def process_questionnaire(path, **retrieval_options):
    """Generate new drafts without changing saved reviews or using assessment as a gate."""
    from dataclasses import asdict
    from trustquery import pipeline
    chunks = retrieval_options['chunks']
    drafts = validate_input(path)
    limit = retrieval_options.get('limit')
    if limit is not None and limit <= 0:
        raise ValueError('Require a positive limit')
    records = []
    for draft in drafts[:limit]:
        record = asdict(draft)
        result = pipeline.answer_question(draft.question, retrieval_options, question_id=draft.id)
        record.update(result)
        record['evidence_assessment'] = dict(implementation_required=result['needs_implementation_evidence'], classification='draft_for_review', reasons=result['reasons'])
        failed = any(reason.startswith('Retrieval failed:') for reason in result['reasons'])
        record.update(review_status='requires_human_review', operational_proof=False,
            errors=[dict(stage='retrieval', message=result['reasons'][-1])] if failed else [],
            outcome='processing_error' if failed else 'policy_extract_draft' if result['answer'] else 'abstained',
            status='pending' if result['answer'] else 'needs_evidence',
            review_note='Draft requires human review.', retrieved_passages=[], evidence_references=[])
        reference_chunks = chunks
        cache_chunks = retrieval_options.get('demo_cache_chunks')
        if cache_chunks is not None and any(reason.startswith('Generation: cached draft') for reason in result['reasons']):
            reference_chunks = cache_chunks
        for c in reference_chunks:
            if any(' '.join(citation['quote'].split()) in ' '.join(c['text'].split()) and citation['source'] == c.get('metadata', {}).get('source') and citation['page'] == c.get('metadata', {}).get('page') for citation in result['citations']):
                m = c['metadata']
                passage = dict(text=c['text'], source=m['source'], page=m['page'],
                    chunk_index=m.get('chunk_index', 0), chunk_id=f"{m['source']}:p{m['page']}:c{m.get('chunk_index', 0)}",
                    evidence_type='policy_requirement', operational_proof=False)
                record['retrieved_passages'].append(passage)
                record['evidence_references'].append(dict(passage, quote=c['text'], role='candidate_for_review'))
        record['provenance'].update(result_method='pipeline_draft',
            generation_method='; '.join(r for r in result['reasons'] if r.startswith(('Generation:', 'Demo cache'))),
            experimental_nli_enabled=False, citations_status='unverified',
            retrieval_method='; '.join(r for r in result['reasons'] if r.startswith('Retrieval:')))
        records.append(record)
    return records


def summarize(records):
    return dict(questions=len(records), outcomes=dict(Counter(r['outcome'] for r in records)),
                errors=sum(bool(r['errors']) for r in records),
                answers=sum(bool(r['answer']) for r in records),
                citations=sum(len(r['citations']) for r in records),
                requires_human_review=len(records))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--corpus', type=Path, default=Path('data/sample'))
    parser.add_argument('--retrieval', choices=['bm25', 'hybrid'], default='bm25')
    parser.add_argument('--index', type=Path)
    parser.add_argument('--prepare-index', action='store_true')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--candidate-k', type=int, default=10)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args(argv)
    offline_mode()
    try:
        validate_input(args.input)
        if args.output.exists():
            raise ValueError('Output already exists; choose a fresh filename')
        if not args.output.parent.is_dir():
            raise ValueError('Output parent directory must exist')
        if args.top_k <= 0 or args.candidate_k < args.top_k or (args.limit is not None and args.limit <= 0):
            raise ValueError('Require candidate_k >= top_k > 0 and a positive limit')
        if args.retrieval == 'hybrid' and args.index is None:
            raise ValueError('--index is required for hybrid retrieval')
        if args.retrieval == 'bm25' and (args.index or args.prepare_index):
            raise ValueError('Index options require --retrieval hybrid')
        chunks = load_corpus(args.corpus, preserve_layout=True)
        collection = reranker = retrieve = None
        if args.retrieval == 'hybrid':
            try:
                collection = open_index(args.index, chunks, args.prepare_index)
                from trustquery.retrieval.reranker import load_reranker
                reranker = load_reranker(local_files_only=True)
            except (ImportError, OSError, RuntimeError, ValueError):
                collection = reranker = None
        else:
            def retrieve(*, query, chunks, top_k, **unused):
                from trustquery.pipeline import _sparse
                return _sparse(query, chunks, top_k)
        records = process_questionnaire(args.input, chunks=chunks, collection=collection,
            reranker_model=reranker, retrieve=retrieve, top_k=args.top_k,
            candidate_k=args.candidate_k, limit=args.limit, offline=args.offline)
        if args.retrieval == 'bm25':
            for r in records:
                r['provenance'].update(retrieval_method='bm25_lexical',
                    embedding_model=None, reranker_model=None)
        with args.output.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(records, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
    except Exception as exc:
        parser.exit(2, f'error: {exc}\n')
    print(json.dumps(summarize(records), sort_keys=True))
    return 1 if any(r['errors'] for r in records) else 0


if __name__ == '__main__':
    raise SystemExit(main())
