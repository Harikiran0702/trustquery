from pathlib import Path
import tempfile
from unittest.mock import Mock

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from trustquery import pipeline, review_gemini, review_workflow
from trustquery.ui import models

APP = Path(__file__).resolve().parents[1] / 'streamlit_app.py'


def element(app, kind, label):
    return next(item for item in getattr(app, kind) if item.label == label)


@pytest.mark.parametrize('flag,fail,expected', [(None, False, False), ('1', False, True), ('1', True, False)])
def test_startup_model_opt_in(monkeypatch, flag, fail, expected):
    from trustquery.ui import upload

    monkeypatch.delenv('TRUSTQUERY_DOWNLOAD_MODELS', raising=False)
    if flag:
        monkeypatch.setenv('TRUSTQUERY_DOWNLOAD_MODELS', flag)
    download = Mock(side_effect=OSError('private diagnostic') if fail else None)
    load_models = Mock()
    process = Mock()
    monkeypatch.setattr('huggingface_hub.snapshot_download', download)
    monkeypatch.setattr(models, 'models', load_models)
    monkeypatch.setattr(upload, 'process', process)
    models.download_search_models.clear()
    try:
        for _ in range(2):
            app = AppTest.from_file(str(APP)).run()
            element(app, 'button', 'Load sample questionnaire').click().run()
            assert not app.exception and not app.error
            assert process.call_args.args[-1] is expected
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

