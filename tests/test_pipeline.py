import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from trustquery import pipeline

ORIGINAL_HYBRID = pipeline._hybrid


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    monkeypatch.setattr(pipeline, '_hybrid', Mock(side_effect=OSError('missing models')))
    monkeypatch.setattr(pipeline, '_faithfulness', Mock(side_effect=OSError('missing NLI')))


@pytest.fixture
def chunks():
    return [dict(text='Policy must require MFA. Legacy services are exempt.',
                 metadata=dict(source='access_control_policy.pdf', page=1, chunk_index=0))]


def test_empty_and_unrelated(chunks):
    for question, corpus in [('MFA?', []), ('Bananas?', chunks)]:
        result = pipeline.answer_question(question, corpus)
        assert result['answer'] == '' and result['confidence'] == 'none'
        assert result['citations'] == []
        assert set(result) == {'answer', 'citations', 'confidence', 'needs_implementation_evidence', 'reasons'}


def test_operational_is_policy_qualified(chunks, monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'mock-only')
    generator = Mock(return_value='We enforce MFA everywhere and are certified.')
    monkeypatch.setattr(pipeline, '_generate', generator)
    result = pipeline.answer_question('Do you currently require MFA?', chunks)
    assert result['needs_implementation_evidence'] is True
    assert result['answer'] == (
        'Policy-based draft (implementation evidence still required):\n\n' + generator.return_value)
    assert result['confidence'] == 'low'
    assert result['citations'][0] == dict(source='access_control_policy.pdf', page=1, quote=chunks[0]['text'])
    assert any('does not establish implementation' in r for r in result['reasons'])


def test_fallback_and_skipped_check(chunks):
    result = pipeline.answer_question('Does policy require MFA?', chunks)
    assert result['answer'] and result['confidence'] in ('low', 'medium')
    assert any('BM25 fallback' in r for r in result['reasons'])
    assert any('Faithfulness check skipped' in r for r in result['reasons'])


@pytest.mark.parametrize('score, expected', [(1, 'high'), (.2, 'low')])
def test_mocked_generation_and_faithfulness(chunks, monkeypatch, score, expected):
    monkeypatch.setenv('GEMINI_API_KEY', 'mock-only')
    monkeypatch.setattr(pipeline, '_generate', Mock(return_value='The policy requires MFA with legacy exemptions.'))
    monkeypatch.setattr(pipeline, '_faithfulness', Mock(return_value=score))
    monkeypatch.setattr(pipeline, '_hybrid', Mock(return_value=chunks))
    result = pipeline.answer_question('Does policy require MFA?', chunks)
    assert result['answer'] == 'The policy requires MFA with legacy exemptions.'
    assert result['confidence'] == expected
    assert result['needs_implementation_evidence'] is False
    assert any('hybrid with local models' in r for r in result['reasons'])


def test_cache_binding_and_tampering(chunks, tmp_path):
    path = tmp_path/'cache.json'
    question = 'Does policy require MFA?'
    entry = dict(method='grounded_gemini', answer='The policy calls for MFA with exemptions.',
        citations=[dict(source='access_control_policy.pdf', page=1, quote=chunks[0]['text'])])
    path.write_text(json.dumps({question: entry}))
    result = pipeline.answer_question(question, dict(chunks=chunks, cache_path=path))
    assert any('cached draft' in r for r in result['reasons'])
    entry['answer'] = 'The policy requires MFA with legacy exemptions.'
    path.write_text(json.dumps({question: entry}))
    result = pipeline.answer_question(question, dict(chunks=chunks, cache_path=path))
    assert result['answer'] == entry['answer']
    assert result['confidence'] == 'medium'
    entry['citations'][0]['page'] = 999
    path.write_text(json.dumps({question: entry}))
    result = pipeline.answer_question(question, dict(chunks=chunks, cache_path=path))
    assert any('not bound' in r for r in result['reasons'])


def test_builder_all_100_without_provider(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('build_demo_cache', root/'scripts/build_demo_cache.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from trustquery.batch_retrieval import load_corpus
    questions = json.loads((root/'data/eval/clouddesk_questionnaire.json').read_text(encoding='utf-8'))
    monkeypatch.setattr(pipeline, '_generate', Mock(side_effect=AssertionError('API forbidden')))
    cache = module.build_cache(questions, load_corpus(root/'data/sample'), limit=100)
    assert len(cache) == len(questions) == 100
    assert all(e['method'] == 'offline_policy_extract' for e in cache.values())
    assert all(set(e) == {'question', 'method', 'answer', 'citations'} for e in cache.values())


def test_batch_uses_pipeline_without_assessment_gate(tmp_path, monkeypatch, chunks):
    from trustquery.batch_process import process_questionnaire
    from trustquery.review_store import load_batch
    path = tmp_path/'q.json'
    path.write_text(json.dumps([dict(id='q1', question='Do you require MFA?', response_type='yes_no')]))
    result = pipeline.answer_question('Do you require MFA?', dict(chunks=chunks, offline=True))
    mock = Mock(return_value=result)
    monkeypatch.setattr(pipeline, 'answer_question', mock)
    records = process_questionnaire(path, chunks=chunks)
    mock.assert_called_once_with('Do you require MFA?', dict(chunks=chunks), question_id='q1')
    assert records[0]['answer']
    assert records[0]['review_status'] == 'requires_human_review'
    assert records[0]['provenance']['citations_status'] == 'unverified'
    output = tmp_path/'batch.json'
    output.write_text(json.dumps(records))
    assert load_batch(output)[0] == records


def test_existing_grounded_generator_adapter(chunks, monkeypatch):
    from trustquery.generation import answer_generator
    monkeypatch.setenv('GEMINI_API_KEY', 'mock-only')
    llm = Mock(return_value='The policy requires MFA, with legacy exemptions.')
    monkeypatch.setattr(answer_generator, 'generate_text', llm)
    result = pipeline.answer_question('Does policy require MFA?', dict(chunks=chunks, retrieval='bm25'))
    assert llm.call_count == 1
    assert chunks[0]['text'] in llm.call_args.args[0]
    assert result['answer'] == llm.return_value
    assert result['citations'][0]['quote'] == chunks[0]['text']
    assert result['citations'][0]['source'] == 'access_control_policy.pdf'


def test_policy_evidence_requirement_is_not_operating_proof():
    assert pipeline._operational('What evidence does the written access control policy require staff to record?') is False
    assert pipeline._operational('Show the latest backup test evidence under the written policy.') is True


def test_cdq001_shared_retrieval():
    from trustquery.batch_retrieval import load_corpus
    from trustquery.retrieval.bm25_retriever import search_bm25
    chunks = load_corpus(pipeline.ROOT/'data/sample')
    q = json.loads((pipeline.ROOT/'data/eval/clouddesk_questionnaire.json').read_text())[0]['question']
    hits = pipeline._sparse(q, chunks, 5)
    assert hits == search_bm25(q, chunks, 5, normalize_query=True)
    assert (hits[0]['metadata']['source'], hits[0]['metadata']['page']) == ('access_control_policy.pdf', 2)


def test_sentences_headers_duplicates_and_offline(chunks, monkeypatch, tmp_path):
    chunks[0]['text'] = 'SYNTHETIC / DEMO | Example Page 1 of 2 MFA is required. Legacy services are exempt. Unfinished'
    monkeypatch.setenv('GEMINI_API_KEY', 'mock-only')
    generator = Mock(side_effect=AssertionError('No external calls'))
    monkeypatch.setattr(pipeline, '_generate', generator)
    result = pipeline.answer_question('Does policy require MFA?', dict(chunks=chunks + chunks, offline=True, retrieval='bm25'))
    assert 'SYNTHETIC' not in result['answer'] and 'Unfinished' not in result['answer']
    assert 'MFA is required.' in result['answer']
    assert len(result['citations']) == 1
    generator.assert_not_called()
    assert all(len(r.split()) <= 12 for r in result['reasons'])


def test_builder_gemini_opt_in(chunks, monkeypatch):
    spec = importlib.util.spec_from_file_location('builder', pipeline.ROOT/'scripts/build_demo_cache.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv('GEMINI_API_KEY', 'mock-only')
    generator = Mock(return_value='The policy requires MFA.')
    monkeypatch.setattr(pipeline, '_generate', generator)
    cache = module.build_cache([{'question': 'Does policy require MFA?'}], chunks, gemini=True)
    generator.assert_called_once()
    assert next(iter(cache.values()))['method'] == 'grounded_gemini'


def test_config_calls_dotenv_without_exposing_values(monkeypatch):
    import importlib
    import dotenv
    from trustquery import config
    loader = Mock()
    monkeypatch.setattr(dotenv, 'load_dotenv', loader)
    importlib.reload(config)
    loader.assert_called_once_with()


def test_cached_references_are_current_and_never_verified(chunks, tmp_path):
    q = 'Does policy require MFA?'
    path = tmp_path/'cache.json'
    entry = dict(answer='The policy calls for MFA with exemptions.', citations=[
        dict(source='access_control_policy.pdf', page=1, verified=True)])
    path.write_text(json.dumps({q: entry}))
    result = pipeline.answer_question(q, dict(chunks=chunks, cache_path=path, retrieval='bm25'))
    assert 'Policy must require MFA.' in result['answer']
    assert 'verified' not in result['citations'][0]
    entry['citations'][0]['source'] = 'invented.pdf'
    path.write_text(json.dumps({q: entry}))
    result = pipeline.answer_question(q, dict(chunks=chunks, cache_path=path, retrieval='bm25'))
    assert result['answer'] != entry['answer']


def test_confidence_ignores_score_sign(chunks, monkeypatch):
    monkeypatch.setattr(pipeline, '_hybrid', Mock(return_value=[dict(chunks[0], reranker_score=-99)]))
    result = pipeline.answer_question('Does policy require MFA?', dict(chunks=chunks, offline=True))
    assert result['confidence'] == 'medium'


@pytest.mark.parametrize('offline', [False, True])
def test_cli_offline_selection(chunks, monkeypatch, tmp_path, offline):
    from trustquery import batch_process
    source = tmp_path/'q.json'
    source.write_text(json.dumps([dict(id='q', question='Does policy require MFA?', response_type='short_answer')]))
    monkeypatch.setattr(batch_process, 'load_corpus', Mock(return_value=chunks))
    process = Mock(return_value=[])
    monkeypatch.setattr(batch_process, 'process_questionnaire', process)
    args = [str(source), str(tmp_path/'out.json')]
    if offline:
        args.append('--offline')
    assert batch_process.main(args) == 0
    assert process.call_args.kwargs['offline'] is offline


def test_evaluator_metrics_and_missing_models(monkeypatch):
    spec = importlib.util.spec_from_file_location('evaluate', pipeline.ROOT/'scripts/evaluate.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, 'try_to_load_from_cache', Mock(return_value=None))
    assert len(module.load_rows(pipeline.ROOT/'data/eval/clouddesk_questionnaire_answer_key.json', pipeline.ROOT/'data/eval/clouddesk_questionnaire.json')) == 100



def test_layout_headings_and_wrapped_sentences():
    passages = [dict(text='SYNTHETIC / DEMO | Access control\nPage 2 of 2\nAccess reviews\nCloudDesk reviews user access permissions every quarter.\nDepartures\nTerminated employee access is revoked within\n24 hours.\nUnfinished trailing fragment', metadata=dict(source='p.pdf', page=1))]
    answer = pipeline._extract(passages, 'access reviews')
    assert answer == 'The policy text states: CloudDesk reviews user access permissions every quarter.'


def test_extract_does_not_split_filenames_or_decimals():
    text = 'Follow access_control_policy.pdf for access reviews. Retain records for 1.5 years.'
    assert pipeline._extract([dict(text=text, metadata=dict(source='p.pdf', page=1))], 'access reviews records years') == 'The policy text states: ' + text


def test_gemini_prompt_passages_and_citation_limit(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'mock-only')
    chunks = [dict(text=f'Access reviews require approval for group {i}.',
                   metadata=dict(source='access.pdf', page=i // 2 + 1)) for i in range(8)]
    generator = Mock(return_value='The policy requires access reviews and approval.')
    monkeypatch.setattr(pipeline, '_generate', generator)
    question = 'What does policy require for access reviews?'
    options = dict(chunks=chunks, retrieve=lambda **kw: chunks, top_k=8)
    result = pipeline.answer_question(question, options)
    offline_result = pipeline.answer_question(question, dict(options, offline=True))
    supplied = generator.call_args.args[1]
    assert supplied == [chunks[0], chunks[2], chunks[4]]
    assert result['answer'] == generator.return_value
    assert [(r['source'], r['page'], r['quote']) for r in result['citations']] == [
        (p['metadata']['source'], p['metadata']['page'], p['text']) for p in supplied]
    assert result['confidence'] == offline_result['confidence'] == 'medium'
    assert [r for r in result['reasons'] if r.startswith('Retrieval term coverage:')] == [
        r for r in offline_result['reasons'] if r.startswith('Retrieval term coverage:')]


def test_idf_sentence_choice_prefers_rare_subject():
    common = 'Access review records require approval.'
    passages = [dict(text=common + ' Cryptographic rotation requires review.',
                     metadata=dict(source='p.pdf', page=1))]
    passages += [dict(text=common, metadata=dict(source='p.pdf', page=i)) for i in range(2, 8)]
    answer, refs = pipeline._extract_bound(passages, 'cryptographic rotation access review records approval')
    assert answer.startswith('The policy text states: Cryptographic rotation requires review.')
    assert len(refs) <= 3


def test_cache_requires_full_evidence_and_matching_question(chunks, tmp_path):
    path = tmp_path / 'cache.json'
    entry = dict(method='grounded_gemini', question='Different question',
                 answer='The policy requires MFA.', citations=[dict(
                     source='access_control_policy.pdf', page=1, quote=chunks[0]['text'])])
    path.write_text(json.dumps({'q': entry}))
    result = pipeline.answer_question('Does policy require MFA?',
        dict(chunks=chunks, question_id='q', cache_path=path))
    assert result['answer'] != entry['answer']
    entry['question'] = 'Does policy require MFA?'
    entry['citations'][0]['quote'] = 'Policy must require MFA.'
    path.write_text(json.dumps({'q': entry}))
    result = pipeline.answer_question(entry['question'], dict(chunks=chunks, question_id='q', cache_path=path))
    assert any('not bound' in r for r in result['reasons'])


def test_normative_gemini_reply_is_not_required_to_say_policy(chunks, monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'mock-only')
    monkeypatch.setattr(pipeline, '_generate', Mock(return_value='Multifactor authentication, with legacy exemptions.'))
    result = pipeline.answer_question('Does policy require MFA?', chunks)
    assert result['answer'] == 'Multifactor authentication, with legacy exemptions.'


def test_live_failure_uses_bound_gemini_cache(chunks, tmp_path, monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'mock-only')
    monkeypatch.setattr(pipeline, '_generate', Mock(side_effect=OSError('mock provider unavailable')))
    question = 'Does policy require MFA?'
    path = tmp_path / 'cache.json'
    answer = 'The policy requires MFA with legacy exemptions.'
    path.write_text(json.dumps({question: dict(method='grounded_gemini', answer=answer,
        citations=[dict(source='access_control_policy.pdf', page=1, quote=chunks[0]['text'])])}))
    result = pipeline.answer_question(question, dict(chunks=chunks, cache_path=path))
    assert result['answer'] == answer
    assert result['confidence'] == 'medium'


def test_operational_gemini_preserves_text_with_evidence_flag(chunks, monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'mock-only')
    answer = 'The policy requires MFA with legacy exemptions.'
    prefix = 'Policy-based draft (implementation evidence still required):'
    monkeypatch.setattr(pipeline, '_generate', Mock(return_value=prefix + '\n\n' + answer))
    monkeypatch.setattr(pipeline, '_faithfulness', Mock(return_value=1))
    result = pipeline.answer_question('Do you require MFA?', chunks)
    assert result['answer'] == prefix + '\n\n' + answer
    assert pipeline._faithfulness.call_args.args[0] == answer
    assert result['needs_implementation_evidence'] is True
    assert result['confidence'] == 'low'


def test_operational_extract_has_at_most_three_complete_sentences():
    chunks = [dict(text='Access reviews require approval. Access reviews require records. Access reviews require owners.',
                   metadata=dict(source='access.pdf', page=1))]
    result = pipeline.answer_question('Do you require access reviews?', dict(chunks=chunks, offline=True, retrieval='bm25'))
    assert len(pipeline._sentences(result['answer'])) == 3
    assert result['answer'].count('.') == 3


@pytest.mark.parametrize('reply, reason', [('', 'empty Gemini response'),
    ('  iNsUfFiCiEnT\n EVIDENCE.', 'Gemini reported insufficient evidence')])
def test_gemini_fallback_explains_reply(chunks, tmp_path, monkeypatch, reply, reason):
    monkeypatch.setenv('GEMINI_API_KEY', 'mock-only')
    monkeypatch.setattr(pipeline, '_generate', Mock(return_value=reply))
    result = pipeline.answer_question('Does policy require MFA?',
        dict(chunks=chunks, cache_path=tmp_path/'missing.json'))
    assert 'Policy must require MFA.' in result['answer']
    assert any(reason in r for r in result['reasons'])
    assert any('Using policy extract' in r for r in result['reasons'])


@pytest.mark.parametrize('cached', [True, False])
def test_default_retrieval_uses_local_hybrid_or_falls_back(chunks, monkeypatch, cached):
    from trustquery import general_rag

    chunk = dict(text='Policy requires MFA for all staff.',
                 metadata=dict(source='access_control_policy.pdf', page=2))
    monkeypatch.setattr(pipeline, '_hybrid', ORIGINAL_HYBRID)
    ranked = Mock(return_value=([chunk], 'hybrid', None) if cached else
                               ([], 'bm25', 'missing_or_unusable_pinned_local_cache'))
    monkeypatch.setattr(general_rag, 'rank', ranked)
    sparse = Mock(return_value=chunks)
    monkeypatch.setattr(pipeline, '_sparse', sparse)
    corpus = chunks + [chunk]
    result = pipeline.answer_question('Does policy require MFA?', dict(chunks=corpus, offline=True))
    ranked.assert_called_once_with('Does policy require MFA?', corpus, hybrid=True)
    assert sparse.called is not cached
    assert result['citations'][0]['page'] == (2 if cached else 1)



def test_gemini_low_coverage_without_faithfulness(chunks, monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'mock-only')
    original = '  Multifactor authentication.\n'
    monkeypatch.setattr(pipeline, '_generate', Mock(return_value=original))
    result = pipeline.answer_question(
        'What does the policy require for MFA during seasonal employee onboarding sessions?', chunks)
    assert result['answer'] == original
    assert result['confidence'] == 'low'


@pytest.mark.parametrize('message', ['503', 'high demand', 'UNAVAILABLE'])
def test_builder_busy_retries_then_resumes(chunks, tmp_path, monkeypatch, capsys, message):
    from trustquery import gemini_client

    script = Path(__file__).resolve().parents[1] / 'scripts/build_demo_cache.py'
    spec = importlib.util.spec_from_file_location('build_demo_cache', script)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    clock = Mock()
    now = 100
    clock.time.side_effect = lambda: now

    def advance(seconds):
        nonlocal now
        now += seconds

    clock.sleep.side_effect = advance
    monkeypatch.setattr(builder, 'time', clock)
    monkeypatch.setenv('GEMINI_API_KEY', 'mock-only')
    answer = 'The policy requires MFA.'
    busy = OSError(message)
    generate = Mock(side_effect=[answer, busy, busy, busy, busy])
    monkeypatch.setattr(gemini_client, 'generate_text', generate)
    questions = [dict(id='q1', question='Does policy require MFA?'),
                 dict(id='q2', question='When does policy require MFA?')]
    path = tmp_path / 'cache.json'
    builder.build_cache(questions, chunks, gemini=True, output=path)
    assert 'Gemini busy: 1 of 2 done. Try again later.' in capsys.readouterr().out
    assert [call.args[0] for call in clock.sleep.call_args_list] == [13, 30, 60, 120]
    assert generate.call_count == 5
    assert all(call.kwargs == {'max_retries': 1} for call in generate.call_args_list)
    saved = json.loads(path.read_text(encoding='utf-8'))
    assert 'q1' in saved and 'q2' not in saved
    generate.side_effect = None
    generate.return_value = answer
    builder.build_cache(questions, chunks, gemini=True, output=path)
    resumed = json.loads(path.read_text(encoding='utf-8'))
    assert resumed['q1'] == saved['q1']
    assert resumed['q2']['method'] == 'grounded_gemini'
    assert generate.call_count == 6
