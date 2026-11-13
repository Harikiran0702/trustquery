from copy import deepcopy
import os
from pathlib import Path
import re
import secrets
import tempfile

import streamlit as st

from trustquery.review_store import ReviewStore, load_batch
from trustquery.ui.export import render_export
from trustquery.ui.models import download_search_models
from trustquery.ui.review_panel import FIELDS, render_metrics, render_review, values
from trustquery.ui.sidebar import render_navigation, render_paths
from trustquery.ui.upload import render_reopen, render_upload


def collect_drafts():
    # Capture submitted widget values before Streamlit removes widgets not rendered
    # this run. Permanent per-question drafts are independent of widget lifecycle.
    sessions = st.session_state.setdefault('review_sessions', {})
    for session in sessions.values():
        for draft in session['drafts'].values():
            if draft.get('reset_widgets'):
                continue
            for field in FIELDS:
                key = draft['prefix'] + field
                if field != 'experimental_policy' and key in st.session_state:
                    draft['edit'][field] = deepcopy(st.session_state[key])
    dirty_any = any(d['edit'] != values(d['base']) for sess in sessions.values() for d in sess['drafts'].values())
    # A path change can arrive in the same request as an editor change, before
    # the browser has seen disabled=True. Retain the accepted context then too.
    accepted_paths = st.session_state.get('review_paths')
    if dirty_any and accepted_paths:
        st.session_state['review_batch_path'], st.session_state['review_state_path'] = accepted_paths
    return sessions, dirty_any, accepted_paths


def hosted_paths(directory, paths):
    if not paths:
        return '', ''
    batch, database = (Path(path) for path in paths)
    run = batch.parent
    if (run.parent != directory or not re.fullmatch(r'[0-9a-f]{32}', run.name)
            or batch != run / 'results.json' or database != run / 'review.sqlite3'
            or any(path.resolve() != path for path in (directory, run, batch, database))
            or not batch.is_file() or not database.is_file()):
        raise ValueError('Invalid saved run for this session')
    return str(batch), str(database)


def main(*, hosted=False):
    st.set_page_config(page_title='TrustQuery human review', layout='wide')
    st.title('TrustQuery human review')
    st.caption("Drafts security-questionnaire answers from policy documents, with citations, and says so when there's no evidence.")
    st.warning('Demo uses fictional policies.')
    sessions, dirty_any, accepted_paths = collect_drafts()
    demo = hosted and not os.getenv('GEMINI_API_KEY', '').strip()
    hybrid = (not demo and os.getenv('TRUSTQUERY_DOWNLOAD_MODELS') == '1'
              and download_search_models())
    directory = None
    if hosted:
        if 'review_temp_directory' not in st.session_state:
            st.session_state['review_temp_directory'] = tempfile.TemporaryDirectory(
                prefix=f'trustquery-review-{secrets.token_hex(16)}-')
        directory = Path(st.session_state['review_temp_directory'].name).resolve()
    render_upload(dirty_any, accepted_paths, directory, hybrid)
    with st.sidebar.expander('Advanced'):
        render_reopen(dirty_any, directory)
        if not hosted:
            batch_path, state_path = render_paths(dirty_any)
    if hosted:
        try:
            batch_path, state_path = hosted_paths(directory, st.session_state.get('review_paths'))
        except (TypeError, ValueError, OSError, RuntimeError):
            st.error('Invalid saved run for this session')
            return
    if not batch_path or not state_path:
        message = 'Reopen an existing review run above or upload and process a new questionnaire.'
        if not hosted:
            message = 'Reopen an existing review run above, upload and process a new questionnaire, or enter a trusted local batch and separate review database in the sidebar.'
        st.info(message)
        return
    try:
        if Path(batch_path).resolve() == Path(state_path).resolve():
            raise ValueError('Input and state must be separate')
        rows, fingerprint, legacy_hash = load_batch(batch_path, include_legacy_hash=True)
        store = ReviewStore(state_path, fingerprint, legacy_hash=legacy_hash)
    except Exception as exc:
        st.error(str(exc))
        return
    context = str(store.path.resolve()) + ':' + fingerprint
    session = sessions.setdefault(context, dict(drafts={}, active=None, applied=None))
    # Retain the active question in sessions with the old disposition filter.
    if session.get('applied') is not None and len(session['applied']) == 6:
        old = session['applied']
        session['applied'] = (*old[:4], old[-1])
    render_export(rows, store, context, batch_path, state_path, dirty_any,
                  hosted=hosted)
    states = {row['id']: store.current(row) for row in rows}
    render_metrics(rows, states)
    requested = render_navigation(rows, states, context, session)
    render_review(rows, states, store, context, session, requested, hosted=hosted, hybrid=hybrid)


if __name__ == '__main__':
    main()
