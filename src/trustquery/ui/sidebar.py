import streamlit as st

def render_paths(dirty_any):
    batch_path = st.text_input('Batch results JSON', '', key='review_batch_path', disabled=dirty_any)
    state_path = st.text_input('Review database', '', key='review_state_path', disabled=dirty_any)
    st.session_state['review_paths'] = (batch_path, state_path)
    if dirty_any:
        st.caption('Save or discard drafts before changing batch or database. Drafts survive navigation within this browser session.')
    return batch_path, state_path


def render_navigation(rows, states, context, session):
    statuses = ['unreviewed', 'edited', 'approved', 'rejected', 'needs_evidence']
    keys = [context + name for name in ('status', 'domain', 'outcome', 'search', 'nav')]
    if session.pop('reset_controls', False):
        for key, value in zip(keys, session['applied']):
            st.session_state[key] = value
    status = st.sidebar.selectbox('Review status', ['All'] + statuses, key=keys[0])
    domains = sorted({r.get('metadata', {}).get('domain', '') for r in rows})
    domain = st.sidebar.selectbox('Domain', ['All'] + domains, key=keys[1])
    outcome = st.sidebar.selectbox('Pipeline status', ['All', 'Drafted', 'Abstained'], key=keys[2])
    search = st.sidebar.text_input('Search ID or question', key=keys[3])
    visible = [r for r in rows if (status == 'All' or states[r['id']]['status'] == status) and (domain == 'All' or r.get('metadata', {}).get('domain', '') == domain) and (outcome == 'All' or draft_status(r) == outcome) and search.casefold() in (r['id'] + ' ' + r['question']).casefold()]
    st.sidebar.caption(f'{len(visible)} of {len(rows)} questions')
    ids = [r['id'] for r in visible]
    nav_key = keys[4]
    if st.session_state.get(nav_key) not in ids:
        st.session_state[nav_key] = ids[0] if ids else None
    def move(delta):
        pos = ids.index(st.session_state[nav_key])
        st.session_state[nav_key] = ids[max(0, min(len(ids)-1, pos + delta))]
    left, right = st.sidebar.columns(2)
    left.button('Previous', on_click=move, args=(-1,), disabled=not ids)
    right.button('Next', on_click=move, args=(1,), disabled=not ids)
    target = st.sidebar.selectbox('Question', ids, key=nav_key) if ids else None
    requested = (status, domain, outcome, search, target)
    return requested


def draft_status(row):
    return 'Abstained' if row.get('abstained') or not row.get('answer') else 'Drafted'
