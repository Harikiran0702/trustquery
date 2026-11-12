from base64 import b64encode
import sqlite3

import streamlit as st


def render_export(rows, store, context, batch_path, state_path, dirty_any, *, hosted=False):
    # Never retain a prepared export across a new saved revision or batch.
    saved_token = tuple((r['id'], store.current(r)['revision']) for r in rows)
    export_key = (context, saved_token)
    prepared = st.session_state.get('prepared_review_export')
    if prepared is not None and prepared[0] != export_key:
        st.session_state.pop('prepared_review_export', None)
        prepared = None
    if st.button('Prepare reviewed Excel', disabled=dirty_any):
        try:
            from trustquery.review_export import export_bytes
            content = export_bytes(batch_path, state_path)
            st.session_state['prepared_review_export'] = (export_key, content)
            prepared = (export_key, content)
        except (ValueError, OSError, sqlite3.Error) as exc:
            st.error(f'Excel export failed: {exc}')
    if prepared is None:
        return
    mime = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    if hosted:
        if not dirty_any:
            # Send the file over the session websocket. A download button would
            # register it with Streamlit's process-wide HTTP media endpoint.
            payload = b64encode(prepared[1]).decode('ascii')
            st.iframe(
                f'<a download="trustquery-reviewed.xlsx" href="data:{mime};base64,{payload}">'
                'Download reviewed Excel</a>',
                alt='Download reviewed Excel',
            )
        return
    st.download_button('Download reviewed Excel', prepared[1],
        file_name='trustquery-reviewed.xlsx', mime=mime, disabled=dirty_any)
