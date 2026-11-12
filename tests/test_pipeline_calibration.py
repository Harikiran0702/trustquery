import importlib.util
import json
from unittest.mock import Mock
import pytest
from trustquery import pipeline


def builder():
    spec = importlib.util.spec_from_file_location('cache_builder_tests', pipeline.ROOT/'scripts/build_demo_cache.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_threshold_and_contributing_citations():
    chunks = [dict(text='Access reviews occur quarterly. Bananas are yellow.', metadata=dict(source='p.pdf', page=1)),
              dict(text='Unrelated access heading.', metadata=dict(source='q.pdf', page=2))]
    q = 'What access reviews occur quarterly?'
    result = pipeline.answer_question(q, dict(chunks=chunks, offline=True, retrieval='bm25'))
    assert result['citations'] == [dict(source='p.pdf', page=1, quote='Access reviews occur quarterly.')]
    assert 'Bananas' not in result['answer']
    result = pipeline.answer_question('Do access reviews integrate with SAML?', dict(chunks=chunks, offline=True, retrieval='bm25'))
    assert result['answer'] == '' and result['citations'] == [] and result['confidence'] == 'none'
    assert result['reasons'][-1] == 'No relevant policy evidence found'


def setup_live(monkeypatch, module, api):
    from trustquery.generation import answer_generator
    from trustquery import gemini_client
    monkeypatch.setattr(gemini_client, 'generate_text', api)
    def answer(q, options):
        answer_generator.generate_text(q)
        return dict(answer='A policy sentence.', citations=[], reasons=['Generation: grounded Gemini draft.'])
    monkeypatch.setattr(module, 'answer_question', answer)
    monkeypatch.setattr(module.time, 'time', Mock(return_value=100))
    monkeypatch.setattr(module.time, 'sleep', Mock())


def test_builder_resume_spacing_atomic_and_counts(tmp_path, monkeypatch, capsys):
    module = builder()
    api = Mock(return_value='mock')
    setup_live(monkeypatch, module, api)
    path = tmp_path/'cache.json'
    rows = [dict(id=f'q{i}', question=f'Question {i}') for i in range(3)]
    cache = module.build_cache(rows, [], gemini=True, output=path, limit=2)
    assert api.call_count == 2
    assert module.time.sleep.call_args.args[0] >= 13
    assert cache['__gemini_usage__']['calls'] == 2
    assert json.loads(path.read_text()) == cache
    module.build_cache(rows, [], gemini=True, output=path, ids={'q0', 'q2'})
    assert api.call_count == 3
    assert 'Total Gemini calls: 1' in capsys.readouterr().out
    assert not list(tmp_path.glob('*.tmp'))


def test_builder_quota_preserves_results_and_daily_cap(tmp_path, monkeypatch, capsys):
    module = builder()
    api = Mock(side_effect=['mock', RuntimeError('429 quota')])
    setup_live(monkeypatch, module, api)
    path = tmp_path/'cache.json'
    rows = [dict(id=f'q{i}', question=f'Question {i}') for i in range(3)]
    module.build_cache(rows, [], gemini=True, output=path)
    cache = json.loads(path.read_text())
    assert 'q0' in cache and 'q1' not in cache
    assert cache['__gemini_usage__']['calls'] == 2
    assert 'Daily quota reached: 1 of 3 done. Run again tomorrow.' in capsys.readouterr().out
    cache['__gemini_usage__']['calls'] = 20
    module.atomic_save(path, cache)
    module.build_cache(rows, [], gemini=True, output=path)
    assert api.call_count == 2
    assert 'Daily quota reached: 0 of 2 done. Run again tomorrow.' in capsys.readouterr().out


def test_builder_other_failure_not_retried(tmp_path, monkeypatch):
    module = builder()
    api = Mock(side_effect=ValueError('bad request'))
    setup_live(monkeypatch, module, api)
    with pytest.raises(ValueError, match='bad request'):
        module.build_cache([dict(id='q', question='q')], [], gemini=True, output=tmp_path/'cache.json')
    assert api.call_count == 1


def test_atomic_replace_failure_preserves_previous(tmp_path, monkeypatch):
    module = builder()
    path = tmp_path/'cache.json'
    path.write_text('{"old": true}')
    monkeypatch.setattr(module.os, 'replace', Mock(side_effect=OSError('mock IO failure')))
    with pytest.raises(OSError):
        module.atomic_save(path, {'new': True})
    assert json.loads(path.read_text()) == {'old': True}
    assert not list(tmp_path.glob('*.tmp'))


def test_builder_nli_backend_loaded_once(monkeypatch):
    module = builder()
    from trustquery.evaluation import nli
    loader = Mock(return_value=object())
    monkeypatch.setattr(nli, 'load_nli_backend', loader)
    def answer(q, options):
        assert options['nli_backend_factory']() is options['nli_backend_factory']()
        return dict(answer='', citations=[], reasons=[])
    monkeypatch.setattr(module, 'answer_question', answer)
    module.build_cache([dict(id='a', question='a'), dict(id='b', question='b')], [], gemini=True)
    loader.assert_called_once_with(local_files_only=True)

def test_evaluator_uses_candidate_ranks_and_metric_denominators(monkeypatch):
    spec = importlib.util.spec_from_file_location('evaluation_metrics_tests', pipeline.ROOT/'scripts/evaluate.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    hits = [dict(metadata=dict(source='other.pdf', page=1)),
            dict(metadata=dict(source='expected.pdf', page=2))]
    monkeypatch.setattr(module, '_sparse', Mock(return_value=hits))
    monkeypatch.setattr(module, 'answer_question', Mock(side_effect=[
        dict(answer='draft', citations=[{}, {}]), dict(answer='', citations=[])]))
    rows = [dict(question='a', classification='policy-supported', should_answer=True, expected_source='expected.pdf', expected_page=2),
            dict(question='b', classification='unsupported/TBD', should_answer=False)]
    m = module.evaluate(rows, [])
    assert (m['page_hit1'], m['page_hit3'], m['MRR'], m['document_hit5']) == (0, 1, .5, 1)
    assert m['abstention_precision'] == m['abstention_recall'] == 1
    assert m['avg_citations_per_nonempty_answer'] == 2


def test_id_keyed_cache_runtime_lookup(tmp_path, monkeypatch):
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    monkeypatch.setattr(pipeline, '_faithfulness', Mock(side_effect=OSError()))
    question = 'Does policy require MFA?'
    chunks = [dict(text='Policy must require MFA.', metadata=dict(source='p.pdf', page=1))]
    path = tmp_path/'cache.json'
    path.write_text(json.dumps({'q1': dict(question=question, method='grounded_gemini',
        answer='Policy must require MFA.', citations=[dict(source='p.pdf', page=1, quote='Policy must require MFA.')])}))
    result = pipeline.answer_question(question, dict(chunks=chunks, retrieval='bm25', cache_path=path))
    assert any('cached draft' in r for r in result['reasons'])
    assert result['citations'][0]['quote'] == 'Policy must require MFA.'


def test_unseen_explicit_negative_and_numeric_subject():
    chunks = [dict(text='Federated login is not supported. Password changes occur on compromise. Exceptions expire within 90 days.',
                   metadata=dict(source='identity.pdf', page=1))]
    for question in ('Is federated login supported?', 'Do password changes occur every 90 days?'):
        result = pipeline.answer_question(question, dict(chunks=chunks, offline=True, retrieval='bm25'))
        assert (result['answer'], result['citations'], result['confidence']) == ('', [], 'none')


def test_acronyms_present_in_evidence_are_not_missing():
    chunks = [dict(text='TLS connections require encryption.', metadata=dict(source='transport.pdf', page=1))]
    result = pipeline.answer_question('What does policy require for TLS connections?',
                                     dict(chunks=chunks, offline=True, retrieval='bm25'))
    assert result['answer'] and result['citations']


def test_unseen_evaluator_aggregate_schema_with_fabricated_rows(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('aggregate_metrics_tests', pipeline.ROOT/'scripts/evaluate.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / 'fabricated.json'
    path.write_text(json.dumps([
        dict(question='Fabricated A?', metadata=dict(should_answer=True)),
        dict(question='Fabricated B?', metadata=dict(should_answer=False)),
        dict(question='Invalid label?', metadata=dict(should_answer='false')),
        None]))
    answer = Mock(side_effect=[dict(answer='draft'), dict(answer='')])
    monkeypatch.setattr(module, 'answer_question', answer)
    assert module.evaluate_unseen(path, []) == dict(records=4, evaluated=2, unrecognized=2,
        should_answer=1, should_abstain=1, drafted=1, abstained=1,
        answerable_drafted=1, nonanswerable_abstained=1)
    assert answer.call_count == 2
    path.write_text('{}')
    assert module.evaluate_unseen(path, []) == dict(status='unrecognized_schema')
