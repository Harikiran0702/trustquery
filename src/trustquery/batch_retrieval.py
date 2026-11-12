"""Offline retrieval and provisional lexical assessment for blank policy drafts."""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path

from trustquery.batch import create_drafts, QuestionnaireError
from trustquery.batch_assessment import assess

POLICIES = (
    'access_control_policy', 'encryption_key_management_policy',
    'information_security_policy', 'incident_response_plan',
    'backup_disaster_recovery_policy', 'change_management_policy',
    'hr_security_policy', 'vendor_risk_management_policy',
)


def offline_mode():
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'


def validate_input(path):
    if 'answer_key' in Path(path).name.lower():
        raise QuestionnaireError('Evaluation answer keys are forbidden runtime input')
    return create_drafts(path)


def load_corpus(directory, *, preserve_layout=False):
    """Read only the eight named PDFs, in fixed order; never evaluation annotations."""
    from trustquery.ingestion.loader import load_pdf
    from trustquery.ingestion.chunker import chunk_documents
    documents = []
    for policy in POLICIES:
        pages = load_pdf(str(Path(directory) / f'{policy}.pdf'), preserve_layout=preserve_layout)
        if not pages:
            raise ValueError(f'Policy has no searchable pages: {policy}')
        documents.extend(pages)
    return chunk_documents(documents)


def identity(chunk):
    metadata = chunk.get('metadata') or {}
    return (metadata.get('source'), metadata.get('page'),
            metadata.get('chunk_index'), chunk.get('text'))


def retrieve_drafts(path, *, chunks, collection, reranker_model,
                    retrieve=None, top_k=5, candidate_k=10, limit=None):
    """Return new draft dictionaries. Retrieval errors remain explicit per question."""
    if top_k <= 0 or candidate_k < top_k or (limit is not None and limit <= 0):
        raise ValueError('Require candidate_k >= top_k > 0 and a positive limit')
    drafts = validate_input(path)  # Validate the entire input before applying limit.
    if retrieve is None:
        offline_mode()
        from trustquery.retrieval.hybrid_retriever import retrieve_and_rerank
        retrieve = retrieve_and_rerank
    known = {identity(chunk) for chunk in chunks}
    fingerprint = hashlib.sha256(json.dumps(chunks, sort_keys=True).encode()).hexdigest()
    output = []
    for draft in drafts[:limit]:
        record = asdict(draft)
        record['retrieved_passages'] = []
        state = 'empty'
        try:
            results = retrieve(query=draft.question, collection=collection, chunks=chunks,
                               reranker_model=reranker_model, top_k=top_k,
                               candidate_k=candidate_k)
            passages = []
            for result in results[:top_k]:
                source, page, index, text = identity(result)
                if (identity(result) not in known or source not in
                        {f'{p}.pdf' for p in POLICIES} or
                        type(page) is not int or page < 1 or
                        type(index) is not int or index < 0 or not text):
                    raise ValueError('Retrieved passage has missing or ungrounded source locators')
                scores = {}
                for key in ('distance', 'bm25_score', 'rrf_score', 'reranker_score'):
                    if key in result:
                        score = float(result[key])
                        if not math.isfinite(score):
                            raise ValueError('Non-finite ranking score')
                        scores[key] = score
                passages.append(dict(source=source, policy_id=Path(source).stem,
                    page=page, chunk_index=index, chunk_id=f'{source}:p{page}:c{index}',
                    text=text, rank=len(passages)+1, scores=scores,
                    evidence_type='synthetic_policy_requirement',
                    assessment_status='unassessed', operational_proof=False))
            record['retrieved_passages'] = passages
            state = 'retrieved_unassessed' if passages else 'empty'
        except Exception as exc:
            state = 'retrieval_error'
            record['retrieval_error'] = {'type': type(exc).__name__, 'message': str(exc)}
        record['status'] = 'pending' if state == 'retrieved_unassessed' else 'needs_evidence'
        record['provenance'].update(evidence_status=state, corpus_sha256=fingerprint,
            retrieval_method='hybrid_rrf_cross_encoder', top_k=top_k, candidate_k=candidate_k,
            embedding_model='all-MiniLM-L6-v2',
            reranker_model='cross-encoder/ms-marco-MiniLM-L-6-v2',
            evidence_boundary='Policy requirements are not proof of implemented controls')
        record['evidence_assessment'] = assess(record['question'], record['retrieved_passages'])
        if record['evidence_assessment']['classification'] != 'policy_supported':
            record['status'] = 'needs_evidence'
        output.append(record)
    return output


def open_index(directory, chunks, prepare=False):
    """Reuse an exact corpus index; never clear or overwrite an existing collection."""
    offline_mode()
    import chromadb
    from trustquery.embeddings.embedder import embed_texts
    from trustquery.vectorstore.chroma_store import add_chunks
    fingerprint = hashlib.sha256(json.dumps(chunks, sort_keys=True).encode()).hexdigest()
    client = chromadb.PersistentClient(path=str(directory))
    name = 'batch_clouddesk_v1'
    names = [item if isinstance(item, str) else item.name for item in client.list_collections()]
    if name not in names:
        if not prepare:
            raise ValueError('No batch index; use --prepare-index once with cached models')
        embeddings = embed_texts([chunk['text'] for chunk in chunks])
        collection = client.create_collection(name, metadata={
            'corpus_sha256': fingerprint, 'embedding_model': 'all-MiniLM-L6-v2'})
        add_chunks(collection, chunks, embeddings)
    else:
        collection = client.get_collection(name)
    if collection.metadata != {'corpus_sha256': fingerprint, 'embedding_model': 'all-MiniLM-L6-v2'}:
        raise ValueError('Index corpus/model mismatch; choose a fresh --index directory')
    stored = collection.get(include=['documents', 'metadatas'])
    indexed = [{'text': text, 'metadata': metadata} for text, metadata in
               zip(stored['documents'], stored['metadatas'])]
    if len(indexed) != len(chunks) or {identity(c) for c in indexed} != {identity(c) for c in chunks}:
        raise ValueError('Incomplete or mismatched index; choose a fresh --index directory')
    return collection


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--corpus', type=Path, default=Path('data/sample'))
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--prepare-index', action='store_true')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--candidate-k', type=int, default=10)
    args = parser.parse_args(argv)
    # Must precede imports: the existing embedder constructs its model at import time.
    offline_mode()
    try:
        validate_input(args.input)
        if args.output.exists():
            raise ValueError('Output already exists')
        if args.top_k <= 0 or args.candidate_k < args.top_k or (args.limit is not None and args.limit <= 0):
            raise ValueError('Invalid retrieval sizes')
        chunks = load_corpus(args.corpus)
        collection = open_index(args.index, chunks, args.prepare_index)
        from trustquery.retrieval.reranker import load_reranker
        records = retrieve_drafts(args.input, chunks=chunks, collection=collection,
            reranker_model=load_reranker(), top_k=args.top_k,
            candidate_k=args.candidate_k, limit=args.limit)
        with args.output.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(records, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
    except (QuestionnaireError, OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f'error: {exc}\n')
    return 1 if any(r['provenance']['evidence_status'] == 'retrieval_error' for r in records) else 0


if __name__ == '__main__':
    raise SystemExit(main())
