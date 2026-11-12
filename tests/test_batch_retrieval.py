import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from trustquery.batch_retrieval import load_corpus, retrieve_drafts, main


@pytest.fixture
def questionnaire(tmp_path):
    path = tmp_path / 'questions.json'
    path.write_text(json.dumps([dict(id=i, question='Require MFA?', response_type='yes_no',
        metadata={'untrusted': 'YES'}) for i in ('q9', 'q1')]))
    return path


@pytest.fixture
def chunk():
    return {'text': 'Synthetic policy requires MFA.', 'metadata': {
        'source': 'access_control_policy.pdf', 'page': 1, 'chunk_index': 0}}


def run(path, chunk, retrieve, **kwargs):
    return retrieve_drafts(path, chunks=[chunk], collection=object(),
        reranker_model=object(), retrieve=retrieve, **kwargs)


def test_mapping_order_provenance_and_no_fabrication(questionnaire, chunk):
    retrieve = Mock(return_value=[{**chunk, 'rrf_score': .03, 'reranker_score': 8.2,
                                  'distance': .1, 'bm25_score': 2}])
    result = run(questionnaire, chunk, retrieve)
    assert [r['id'] for r in result] == ['q9', 'q1']
    for i, record in enumerate(result, 1):
        assert record['answer'] == '' and record['citations'] == []
        assert record['status'] == 'needs_evidence'
        assert record['evidence_assessment']['classification'] == 'implementation_evidence_required'
        assert record['provenance']['input_record'] == i
        assert record['provenance']['operational_proof'] is False
        passage = record['retrieved_passages'][0]
        assert passage['chunk_id'] == 'access_control_policy.pdf:p1:c0'
        assert passage['policy_id'] == 'access_control_policy'
        assert passage['scores']['reranker_score'] == 8.2
        assert passage['assessment_status'] == 'unassessed'
        assert passage['operational_proof'] is False
    assert retrieve.call_args.kwargs['query'] == 'Require MFA?'
    assert run(questionnaire, chunk, retrieve) == result


@pytest.mark.parametrize('change', [
    {'metadata': {}}, {'metadata': {'source': 'answer_key.json', 'page': 1, 'chunk_index': 0}},
    {'text': 'fabricated'}, {'reranker_score': float('nan')},
])
def test_missing_or_ungrounded_results(questionnaire, chunk, change):
    records = run(questionnaire, chunk, Mock(return_value=[{**chunk, **change}]))
    assert all(r['status'] == 'needs_evidence' and not r['retrieved_passages'] for r in records)
    assert records[0]['provenance']['evidence_status'] == 'retrieval_error'


def test_empty_and_exception_continue(questionnaire, chunk):
    records = run(questionnaire, chunk, Mock(side_effect=[RuntimeError('offline failure'), []]))
    assert [r['provenance']['evidence_status'] for r in records] == ['retrieval_error', 'empty']
    assert all(r['status'] == 'needs_evidence' and r['answer'] == '' and r['citations'] == [] for r in records)


def test_limit_and_entire_input_validation(questionnaire, chunk):
    retrieve = Mock(return_value=[])
    assert len(run(questionnaire, chunk, retrieve, limit=1)) == 1
    assert retrieve.call_count == 1
    questionnaire.write_text(json.dumps([dict(id='ok', question='Q', response_type='yes_no'), {}]))
    with pytest.raises(ValueError):
        run(questionnaire, chunk, retrieve, limit=1)
    assert retrieve.call_count == 1


@pytest.mark.parametrize('kwargs', [{'limit': 0}, {'top_k': 0}, {'top_k': 11}])
def test_bad_sizes(questionnaire, chunk, kwargs):
    with pytest.raises(ValueError):
        run(questionnaire, chunk, Mock(), **kwargs)


def test_eight_policy_corpus_and_sparse_integration():
    from trustquery.retrieval.bm25_retriever import search_bm25
    root = Path(__file__).resolve().parents[1]
    chunks = load_corpus(root / 'data/sample')
    assert len({c['metadata']['source'] for c in chunks}) == 8
    assert chunks == load_corpus(root / 'data/sample')
    results = search_bm25('multi-factor authentication', chunks, top_k=3)
    assert results and all(r['metadata']['page'] > 0 for r in results)


def test_missing_policy(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path)


def test_cli_invalid_input_no_output(tmp_path):
    path = tmp_path / 'answer_key.json'
    path.write_text('{"answers": []}')
    output = tmp_path / 'out.json'
    with pytest.raises(SystemExit) as exc:
        main([str(path), str(output), '--index', str(tmp_path / 'index')])
    assert exc.value.code == 2 and not output.exists()


def test_answer_key_rejected_before_read(questionnaire, chunk, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('must not read answer key')
    monkeypatch.setattr(Path, 'read_bytes', forbidden)
    with pytest.raises(ValueError, match='forbidden'):
        run(questionnaire.with_name('clouddesk_questionnaire_answer_key.json'), chunk, Mock())


def test_index_reuse_and_mismatch(chunk, monkeypatch):
    import hashlib
    import chromadb
    from trustquery.batch_retrieval import open_index
    fingerprint = hashlib.sha256(json.dumps([chunk], sort_keys=True).encode()).hexdigest()
    collection = Mock()
    collection.metadata = {'corpus_sha256': fingerprint, 'embedding_model': 'all-MiniLM-L6-v2'}
    collection.get.return_value = {'documents': [chunk['text']], 'metadatas': [chunk['metadata']]}
    client = Mock()
    client.list_collections.return_value = ['batch_clouddesk_v1']
    client.get_collection.return_value = collection
    monkeypatch.setattr(chromadb, 'PersistentClient', Mock(return_value=client))
    import trustquery.embeddings.embedder as embedder
    embed = Mock(side_effect=AssertionError('must reuse embeddings'))
    monkeypatch.setattr(embedder, 'embed_texts', embed)
    assert open_index('unused', [chunk], prepare=True) is collection
    client.create_collection.assert_not_called()
    embed.assert_not_called()
    collection.metadata = {}
    with pytest.raises(ValueError, match='mismatch'):
        open_index('unused', [chunk])
    client.list_collections.return_value = []
    with pytest.raises(ValueError, match='prepare-index'):
        open_index('unused', [chunk])
