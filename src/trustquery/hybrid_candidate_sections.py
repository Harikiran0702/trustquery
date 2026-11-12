"""Opt-in offline hybrid section locators. Always abstains; never writes."""
import argparse
from functools import lru_cache
import json
import math
import re

from trustquery import candidate_sections as base
from trustquery.model_settings import EMBEDDING_REVISION as EMBED_REVISION, RERANKER_REVISION as RERANK_REVISION
from trustquery.retrieval.rrf import reciprocal_rank_fusion

EMBEDDER = 'sentence-transformers/all-MiniLM-L6-v2'
RERANKER = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
CANDIDATE_K = 10
# Explicit corpus scopes prevent semantic similarity from resolving scope collisions.
SCOPES = {
 'vendor_risk_management_policy.pdf': r'vendor|supplier',
 'hr_security_policy.pdf': r'personnel|worker|employee|hr',
 'incident_response_plan.pdf': r'incident',
 'change_management_policy.pdf': r'change|deployment|rollback',
 'backup_disaster_recovery_policy.pdf': r'backup|restore|restoration|recovery',
 'encryption_key_management_policy.pdf': r'encryption|decrypt|cryptographic',
 'information_security_policy.pdf': r'information security',
}


@lru_cache(maxsize=1)
def models():
    # No existing embedder import: that module initializes a network-capable model.
    from huggingface_hub import try_to_load_from_cache
    for repo, revision in ((EMBEDDER, EMBED_REVISION), (RERANKER, RERANK_REVISION)):
        if not isinstance(try_to_load_from_cache(repo, 'config.json', revision=revision), str):
            raise OSError(f'Pinned local model unavailable: {repo}')
    from sentence_transformers import SentenceTransformer, CrossEncoder
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    encoder = SentenceTransformer(EMBEDDER, revision=EMBED_REVISION,
        local_files_only=True, device='cpu')
    reranker = CrossEncoder(RERANKER, revision=RERANK_REVISION,
        local_files_only=True, device='cpu', max_length=512)
    encoder.eval()
    reranker.model.eval()
    return encoder, reranker


def rank(question, sections):
    """Search all freshly verified sections; no persistent or stale vector index."""
    import numpy as np
    from trustquery.retrieval.reranker import rerank_chunks
    encoder, reranker = models()
    vectors = encoder.encode([question] + [s['text'] for s in sections],
        normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    scores = vectors[1:] @ vectors[0]
    if not np.isfinite(scores).all():
        raise ValueError('Non-finite dense score')
    dense = [dict(s, dense_score=float(scores[i])) for i,s in enumerate(sections)]
    dense.sort(key=lambda h: -h['dense_score'])
    sparse = base.search_bm25(question, sections, top_k=len(sections))
    fused = reciprocal_rank_fusion(dense[:CANDIDATE_K], sparse[:CANDIDATE_K])
    for h in fused:
        key = h['metadata']
        for name, hits in [('dense_rank',dense), ('bm25_rank',sparse)]:
            h[name] = next(i for i,x in enumerate(hits,1) if x['metadata']==key)
    selected = fused[:CANDIDATE_K]
    for i,h in enumerate(selected,1):
        h['rrf_rank'] = i
    ranked = rerank_chunks(question, selected, reranker, top_k=len(selected))
    if any(not math.isfinite(h['reranker_score']) for h in ranked):
        raise ValueError('Non-finite reranker score')
    return ranked


def locate(record, *, enabled=False):
    # Reuse unchanged eligibility, lexical scope guards and atomic corpus checks.
    out = base.locate(record, enabled=enabled)
    out.update(method='offline_minilm_bm25_rrf_crossencoder_sections_v1', candidates=[])
    out.pop('plausible_section_count', None)
    out['locator_status'] = 'ambiguous_or_missing_locator'
    if 'corpus_sha256' not in out:
        return out
    question = record['question']
    scopes = [s for s,p in SCOPES.items() if re.search(r'\b(?:'+p+r')\b',question,re.I)]
    if len(scopes) != 1:
        return dict(out, reason='Missing or multiple explicit policy scopes; abstained')
    try:
        sections, excluded, hashes = base.load_sections()
        identities = [(s['metadata']['source'],s['metadata']['page']) for s in sections]
        if len(set(identities)) != len(identities):
            raise ValueError('Section identity collision')
        by_id = dict(zip(identities,sections))
        ranked = rank(question, sections)
        candidates = []
        trace = []
        for i,h in enumerate(ranked,1):
            key = h['metadata']['source'],h['metadata']['page']
            section = by_id[key]
            if h['text'] != section['text']:
                raise ValueError('Ranked section binding mismatch')
            overlap = sorted(base.terms(question) & base.terms(section['text']))
            # Rank scores only order candidates, never grant support or uniqueness.
            display = key[0] == scopes[0] and len(overlap) >= base.MIN_OVERLAP
            trace.append(dict(source=key[0],page=key[1],rerank_rank=i,
                dense_rank=h['dense_rank'],bm25_rank=h['bm25_rank'],rrf_rank=h['rrf_rank'],
                displayed=display and len(candidates)<base.MAX_CANDIDATES))
            if display and len(candidates)<base.MAX_CANDIDATES:
                ref=section['document_context'][key[1]-1]
                candidates.append(dict(source=key[0],page=key[1],section_number=key[1],
                    title=section['title'],section_text=section['text'],
                    section_utf8_sha256=base.sha(section['text'].encode()),
                    pdf_sha256=ref['pdf_sha256'],page_utf8_sha256=ref['page_utf8_sha256'],
                    document_context=section['document_context'],overlap=overlap,
                    score=h['reranker_score'],dense_score=h.get('dense_score'),
                    rrf_score=h['rrf_score'],ranks=trace[-1],
                    evidence_status='UNVERIFIED CANDIDATE'))
        # Detect source changes during model inference before publishing any text.
        if base.load_sections()[2] != hashes:
            raise ValueError('Source version changed during ranking')
        out.update(candidates=candidates, ranking_trace=trace, corpus_sha256=hashes,
            excluded_layouts=excluded, locator_status='unadjudicated_similarity_candidates',
            reason='Candidates only; always abstains. Scores are uncalibrated relevance, not '
                'entailment, completeness, contradiction resolution or verified citations. '
                'Explicit scope and lexical gates over-refuse; human review required.',
            model_versions=dict(embedding=EMBED_REVISION,reranker=RERANK_REVISION))
    except Exception as exc:
        out.update(candidates=[], reason='Offline hybrid unavailable/integrity failure: '+str(exc))
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('question')
    parser.add_argument('--enable-hybrid-section-candidates',action='store_true')
    args=parser.parse_args()
    print(json.dumps(locate(dict(question=args.question,outcome='abstained',
        evidence_assessment={'implementation_required':False}),
        enabled=args.enable_hybrid_section_candidates),indent=2))

if __name__ == '__main__':
    main()
