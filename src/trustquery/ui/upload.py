from pathlib import Path
import sqlite3

import streamlit as st


SAMPLE = Path(__file__).resolve().parents[3] / 'data/eval/clouddesk_questionnaire.json'


def select_paths(batch, database):
    st.session_state['review_batch_path'] = batch
    st.session_state['review_state_path'] = database
    st.session_state['review_paths'] = (batch, database)


def process(name, read_bytes, directory, hybrid):
    from trustquery.review_workflow import create_run

    try:
        with st.spinner('Retrieving policy evidence offline...'):
            batch, database, count = create_run(name, read_bytes(), directory=directory,
                offline=True, hybrid=hybrid, cache_ranking=directory is None)
        select_paths(batch, database)
        st.success(f'Processed {count} questions.')
    except (ValueError, OSError, sqlite3.Error, RecursionError) as exc:
        st.error(f'Processing failed: {exc}. Existing reviews are retained.')


def render_upload(dirty_any, accepted_paths, directory, hybrid):
    with st.expander('Start a questionnaire review', expanded=not bool(accepted_paths and accepted_paths[0])):
        if st.button('Load sample questionnaire', disabled=dirty_any):
            process(SAMPLE.name, SAMPLE.read_bytes, directory, hybrid)
        st.download_button('Download questionnaire template', b'id,question,response_type\nq1,What does the written access control policy require for access reviews?,short_answer\n', file_name='questionnaire-template.csv', mime='text/csv')
        upload = st.file_uploader('Upload security questionnaire', type=['csv', 'json'], disabled=dirty_any)
        valid = False
        if upload is not None:
            try:
                from trustquery.review_workflow import validate_upload
                _, count = validate_upload(upload.name, upload.getvalue())
                st.success(f'Validated {count} questions. Processing creates a new isolated review run.')
                valid = True
            except (ValueError, OSError, RecursionError) as exc:
                st.error(f'Invalid questionnaire: {exc}')
        if st.button('Process questionnaire offline', disabled=dirty_any or not valid):
            process(upload.name, upload.getvalue, directory, hybrid)


def render_reopen(dirty_any, directory):
    st.caption('Reopen existing review run')
    from trustquery.review_workflow import list_run_ids, reopen_run, saved_run_ids
    try:
        run_ids = saved_run_ids() if directory is None else list_run_ids(directory)
    except (ValueError, OSError) as exc:
        st.error(f'Cannot list saved runs: {exc}')
        run_ids = []
    run_id = st.selectbox('Saved review run', [''] + run_ids, disabled=dirty_any,
                          format_func=lambda value: value or 'Choose a saved run')
    if st.button('Reopen selected run', disabled=dirty_any or not run_id):
        try:
            if run_id not in run_ids:
                raise ValueError('Invalid saved run for this session')
            batch, database = reopen_run(run_id, directory)
            select_paths(batch, database)
            st.success('Reopened saved run. Existing decisions and revisions are retained.')
        except (ValueError, OSError, sqlite3.Error) as exc:
            st.error(f'Cannot reopen saved run: {exc}. Existing reviews are retained.')
