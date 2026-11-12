from copy import deepcopy
from pathlib import Path

import streamlit as st

from trustquery.review_store import ReviewStore, load_batch
from trustquery.ui.export import render_export
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


def main():
    st.set_page_config(page_title='TrustQuery human review', layout='wide')
    st.title('TrustQuery human review')
    st.caption("Drafts security-questionnaire answers from policy documents, with citations, and says so when there's no evidence.")
    st.warning('Demo uses fictional policies.')
    sessions, dirty_any, accepted_paths = collect_drafts()
    render_upload(dirty_any, accepted_paths)
    with st.sidebar.expander('Advanced'):
        render_reopen(dirty_any)
        batch_path, state_path = render_paths(dirty_any)
    if not batch_path or not state_path:
        st.info('Reopen an existing review run above, upload and process a new questionnaire, or enter a trusted local batch and separate review database in the sidebar.')
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
    render_export(rows, store, context, batch_path, state_path, dirty_any)
    states = {row['id']: store.current(row) for row in rows}
    render_metrics(rows, states)
    requested = render_navigation(rows, states, context, session)
    render_review(rows, states, store, context, session, requested)


if __name__ == '__main__':
    main()
