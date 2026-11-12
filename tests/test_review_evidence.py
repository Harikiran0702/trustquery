"""Viewer boundaries and session/context isolation; only temporary PDFs/state."""
import json
from pathlib import Path
import pytest
from pypdf import PdfWriter
from trustquery.review_evidence import page_context


def ref(**changes):
    return dict(source='access_control_policy.pdf', page=1, chunk_index=0,
                chunk_id='candidate', quote='example', **changes)


@pytest.mark.parametrize('source', ['../access_control_policy.pdf', '..\\access_control_policy.pdf',
    '/access_control_policy.pdf', 'C:\\private.pdf', 'C:access_control_policy.pdf',
    '\\\\server\\share\\access_control_policy.pdf', 'https://example.com/p.pdf',
    'access_control_policy.pdf:secret', '%2e%2e%2faccess_control_policy.pdf',
    'access_control_policy.pdf.', 'ACCESS_CONTROL_POLICY.PDF', 'unknown.pdf'])
def test_path_manipulation(tmp_path, source):
    r = ref()
    r['source'] = source
    with pytest.raises(ValueError, match='basename'):
        page_context(r, tmp_path)


def test_redirected_file(tmp_path):
    outside = tmp_path/'outside.pdf'
    outside.write_bytes(b'%PDF-')
    link = tmp_path/'access_control_policy.pdf'
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip('Symlink privilege unavailable')
    with pytest.raises(ValueError, match='Redirected'):
        page_context(ref(), tmp_path)


def test_limits_missing_and_malformed(tmp_path):
    with pytest.raises(FileNotFoundError):
        page_context(ref(), tmp_path)
    p = tmp_path/'access_control_policy.pdf'
    p.write_bytes(b'not a PDF')
    with pytest.raises(ValueError, match='not a PDF'):
        page_context(ref(), tmp_path)
    with p.open('wb') as f:
        f.truncate(20*1024*1024+1)
    with pytest.raises(ValueError, match='20 MiB'):
        page_context(ref(), tmp_path)
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.write(p)
    r = ref()
    r['page'] = 2
    with pytest.raises(ValueError, match='page is unavailable'):
        page_context(r, tmp_path)
    for page in (True, 0, -1, '1'):
        r['page'] = page
        with pytest.raises(ValueError, match='one-based'):
            page_context(r, tmp_path)
    writer.encrypt('secret')
    writer.write(p)
    with pytest.raises(ValueError, match='Encrypted'):
        page_context(ref(), tmp_path)


def test_actual_pdf_context_and_mismatch():
    r = ref()
    r['page'] = 2
    r['quote'] = 'CloudDesk reviews user access permissions every quarter.'
    result = page_context(r)
    assert result['quote_present']
    assert [n for n, text in result['pages']] == [1, 2]
    assert result['source'] == r['source'] and result['page'] == 2
    r['quote'] = 'Unrelated operational claim'
    assert not page_context(r)['quote_present']


def test_ui_context_isolation_and_draft_retention(tmp_path, monkeypatch):
    AppTest = pytest.importorskip('streamlit.testing.v1').AppTest
    monkeypatch.chdir(tmp_path)
    batch = tmp_path/'data/batch_runs/major_deliverable_1/results.json'
    batch.parent.mkdir(parents=True)
    rows = [dict(id=f'q{i}', question=f'Question {i}', answer='', outcome='abstained',
                 metadata={}, evidence_assessment={'implementation_required': False},
                 evidence_references=[ref()]) for i in (1, 2)]
    rows[1]['evidence_references'][0]['source'] = '../secret.pdf'
    batch.write_text(json.dumps(rows))
    app_path = Path(__file__).resolve().parents[1]/'src/trustquery/review_app.py'
    app = local_app(AppTest, str(app_path), default_timeout=30).run()
    def element(kind, label):
        return next(e for e in getattr(app, kind) if e.label == label)
    def context_toggle():
        tools = next(item for item in app.expander if item.label == 'Additional review tools')
        return next(toggle for toggle in tools.toggle if toggle.label.startswith('Show local PDF context'))
    element('text_area', 'Answer draft').set_value('Retained draft')
    context_toggle().set_value(True).run()
    assert not app.exception
    assert element('text_area', 'Answer draft').value == 'Retained draft'
    assert any('CURRENT LOCAL PDF' in c.value for c in app.caption)
    element('selectbox', 'Question').select('q2').run()
    assert any('Unsaved edits' in w.value for w in app.warning)
    assert any('CURRENT LOCAL PDF' in c.value for c in app.caption)
    element('button', 'Discard changes and continue').click().run()
    assert not context_toggle().value
    context_toggle().set_value(True).run()
    assert not app.exception
    assert any('context unavailable' in w.value for w in app.warning)
    assert not any('CURRENT LOCAL PDF' in c.value for c in app.caption)
    assert not any(c.value == 'Candidate evidence - unverified' for c in app.subheader)
    assert not list(tmp_path.rglob('*.pdf'))


def local_app(AppTest, *args, **kwargs):
    app = AppTest.from_file(*args, **kwargs)
    app.session_state['review_batch_path'] = 'data/batch_runs/major_deliverable_1/results.json'
    app.session_state['review_state_path'] = 'data/reviews/major_deliverable_2.sqlite3'
    return app
