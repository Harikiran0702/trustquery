import json
from pathlib import Path
import tempfile
from unittest.mock import Mock

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from trustquery import pipeline, review_gemini, review_workflow
from trustquery.ui import app as review_app
from trustquery.ui import models, upload

APP = Path(__file__).resolve().parents[1] / 'streamlit_app.py'


def element(app, kind, label):
    return next(item for item in getattr(app, kind) if item.label == label)


@pytest.fixture(autouse=True)
def clear_sample_cache():
    upload.prepare_sample.clear()
    yield
    upload.prepare_sample.clear()


@pytest.mark.parametrize('flag,fail,expected', [(None, False, False), ('1', False, True), ('1', True, False)])
def test_startup_model_opt_in(monkeypatch, flag, fail, expected):
    monkeypatch.setenv('GEMINI_API_KEY', 'mock-only')
    monkeypatch.delenv('TRUSTQUERY_DOWNLOAD_MODELS', raising=False)
    if flag:
        monkeypatch.setenv('TRUSTQUERY_DOWNLOAD_MODELS', flag)
    download = Mock(side_effect=OSError('private diagnostic') if fail else None)
    load_models = Mock()
    prepare = Mock(return_value=None)
    monkeypatch.setattr('huggingface_hub.snapshot_download', download)
    monkeypatch.setattr(models, 'models', load_models)
    monkeypatch.setattr(upload, 'prepare_sample', prepare)
    models.download_search_models.clear()
    try:
        for _ in range(2):
            app = AppTest.from_file(str(APP)).run()
            assert not app.exception and not app.error
            assert prepare.call_args.args == (expected,)
        assert download.call_count == (0 if flag is None else 1 if fail else 2)
        if expected:
            assert [(c.kwargs['repo_id'], c.kwargs['revision']) for c in download.call_args_list] == [
                (models.EMBEDDER, models.EMBED_REVISION), (models.RERANKER, models.RERANK_REVISION),
            ]
            load_models.assert_called_once_with()
        else:
            load_models.assert_not_called()
    finally:
        models.download_search_models.clear()


def test_sample_is_built_once_and_copied_into_private_runs(monkeypatch):
    from trustquery.review_store import ReviewStore, load_batch

    monkeypatch.setenv('TRUSTQUERY_DOWNLOAD_MODELS', '1')
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    forbidden = Mock(side_effect=AssertionError('Unexpected external call'))
    monkeypatch.setattr(review_app, 'download_search_models', forbidden)
    monkeypatch.setattr(pipeline, '_generate', forbidden)
    monkeypatch.setattr(review_gemini, 'request', forbidden)
    cached = json.loads(pipeline.DEMO_CACHE.read_text(encoding='utf-8'))['cdq001']
    assert cached['method'] == 'grounded_gemini'
    row = {
        'id': 'cdq001', 'question': 'Sample question', 'answer': cached['answer'],
        'outcome': 'answered', 'evidence_assessment': {},
        'confidence': 'high', 'evidence_references': [],
    }

    def create_sample(name, raw, *, offline, hybrid, cache_ranking):
        assert name == upload.SAMPLE.name and raw == upload.SAMPLE.read_bytes()
        assert offline and not hybrid and not cache_ranking
        return json.dumps([row]).encode(), 1

    create = Mock(side_effect=create_sample)
    spinner = Mock(wraps=st.spinner)
    monkeypatch.setattr(review_workflow, 'prepare_run', create)
    monkeypatch.setattr(st, 'spinner', spinner)

    owner = AppTest.from_file(str(APP)).run()
    visitor = AppTest.from_file(str(APP)).run()
    owner.run()
    for app in (owner, visitor):
        assert not app.exception and not app.error, [error.value for error in app.error]
        element(app, 'button', 'Load sample questionnaire').click().run()
        assert not app.exception and not app.error
        assert element(app, 'text_area', 'Answer draft').value == cached['answer']

    owner_batch, owner_db = owner.session_state['review_paths']
    visitor_batch, visitor_db = visitor.session_state['review_paths']
    assert Path(owner_batch).parent != Path(visitor_batch).parent
    assert Path(owner_batch).read_bytes() == Path(visitor_batch).read_bytes()
    element(owner, 'text_area', 'Answer draft').set_value('Private edit').run()
    assert element(visitor.run(), 'text_area', 'Answer draft').value == cached['answer']
    rows, digest = load_batch(visitor_batch)
    owner_store = ReviewStore(owner_db, digest)
    owner_store.save(rows[0], revision=0, action='rejected', answer='Private edit',
                     selected=[], reviewer='Owner', note='Needs revision')
    assert owner_store.current(rows[0])['status'] == 'rejected'
    visitor_state = ReviewStore(visitor_db, digest).current(rows[0])
    assert visitor_state['status'] == 'unreviewed'
    assert visitor_state['answer'] == cached['answer']
    assert Path(owner_batch).parent.name not in element(visitor, 'selectbox', 'Saved review run').options

    element(visitor, 'button', 'Load sample questionnaire').click().run()
    assert visitor.session_state['review_paths'][1] != visitor_db
    create.assert_called_once()
    assert [call.args for call in spinner.call_args_list
            if call.args == ('Preparing sample (first load only)',)] == [
                ('Preparing sample (first load only)',),
            ]
    forbidden.assert_not_called()


def test_hosted_offline_run_is_private(monkeypatch):
    monkeypatch.delenv('TRUSTQUERY_DOWNLOAD_MODELS', raising=False)
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    gemini = Mock(side_effect=AssertionError('Unexpected Gemini call'))
    monkeypatch.setattr(review_gemini, 'request', gemini)
    monkeypatch.setattr(pipeline, '_generate', gemini)
    raw = b'id,question,response_type\nq1,What does the written access control policy require for access reviews?,short_answer\n'

    class Upload:
        name = 'questionnaire.csv'

        def getvalue(self):
            return raw

    monkeypatch.setattr(st, 'file_uploader', lambda *args, **kwargs: Upload())
    prepare = Mock(return_value=None)
    monkeypatch.setattr(upload, 'prepare_sample', prepare)
    owner = AppTest.from_file(str(APP), default_timeout=40).run()
    element(owner, 'button', 'Process questionnaire offline').click().run()
    assert not owner.exception and not owner.error
    assert not any(expander.label == 'Generate a live AI draft' for expander in owner.expander)
    assert not any('gemini' in toggle.label.casefold() for toggle in owner.toggle)
    assert not any(item.label in ('Batch results JSON', 'Review database') for item in owner.text_input)
    batch, database = map(Path, owner.session_state['review_paths'])
    run = batch.parent
    directory = run.parent
    assert directory.parent == Path(tempfile.gettempdir()).resolve()
    assert batch.is_file() and database.is_file() and database.parent == run
    assert run.name in element(owner, 'selectbox', 'Saved review run').options
    element(owner, 'selectbox', 'Saved review run').select(run.name).run()
    element(owner, 'button', 'Reopen selected run').click().run()
    assert not owner.exception and not owner.error
    element(owner, 'button', 'Prepare reviewed Excel').click().run()
    assert owner.session_state['prepared_review_export'][1]
    assert 'base64,' in owner.get('iframe')[0].proto.srcdoc
    assert not any(button.proto.label == 'Download reviewed Excel' for button in owner.get('download_button'))

    visitor = AppTest.from_file(str(APP)).run()
    other = Path(visitor.session_state['review_temp_directory'].name).resolve()
    assert directory != other
    assert run.name not in element(visitor, 'selectbox', 'Saved review run').options
    assert 'prepared_review_export' not in visitor.session_state
    for run_id in (run.name, f'../{directory.name}/{run.name}', str(run)):
        with pytest.raises(ValueError):
            review_workflow.reopen_run(run_id, other)
    for paths in ((str(batch), str(database)),
                  tuple(str(other / '..' / directory.name / run.name / p.name) for p in (batch, database))):
        visitor.session_state['review_paths'] = paths
        visitor.run()
        assert not visitor.exception
        assert any(error.value == 'Invalid saved run for this session' for error in visitor.error)
        assert not any(button.label == 'Prepare reviewed Excel' for button in visitor.button)
        assert 'prepared_review_export' not in visitor.session_state
    local = AppTest.from_file(str(APP.parent / 'src/trustquery/review_app.py'))
    local.session_state['review_batch_path'] = str(batch)
    local.session_state['review_state_path'] = str(database)
    local.run()
    assert not local.exception and not local.error
    assert not any(expander.label == 'Generate a live AI draft' for expander in local.expander)
    assert not any('gemini' in toggle.label.casefold() for toggle in local.toggle)
    gemini.assert_not_called()

