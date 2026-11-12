from copy import deepcopy
from functools import partial
import sqlite3

import streamlit as st

from trustquery.ui.sidebar import draft_status
from trustquery.review_evidence import page_context
from trustquery.review_store import ReviewConflict, policy_extract

FIELDS = ('answer', 'selected', 'reviewer', 'note', 'scope_checked', 'experimental_policy')


def values(state):
    return {field: deepcopy(state.get(field)) for field in FIELDS}


def render_metrics(rows, states):
    labels = ('Drafted', 'Abstained', 'Approved', 'Rejected', 'Needs evidence')
    counts = [sum(draft_status(row) == label for row in rows) for label in labels[:2]]
    counts += [sum(state['status'] == status for state in states.values())
               for status in ('approved', 'rejected', 'needs_evidence')]
    for column, label, count in zip(st.columns(5), labels, counts):
        column.metric(label, count)


def render_review(rows, states, store, context, session, requested):
    target = requested[-1]
    drafts = session['drafts']
    active = session['active']
    pending = active is not None and requested != session['applied'] and drafts[active]['edit'] != values(drafts[active]['base'])
    if not pending:
        session['active'] = target
        session['applied'] = requested
        active = target
    if active is None:
        st.info('No questions match these filters.')
        return
    r = next(r for r in rows if r['id'] == active)
    if active not in drafts:
        base = deepcopy(states[active])
        drafts[active] = dict(base=base, edit=values(base), prefix=context + ':' + active + ':', conflict=None)
    draft = drafts[active]
    prefix = draft['prefix']

    save = partial(save_draft, store, r, draft)
    render_pending(session, requested, store, r, draft, pending, save)
    current = draft['base']
    latest = states[active]
    st.subheader(active + ': ' + r['question'])
    render_draft(r)
    st.caption(f"Review: {current['status']} | Revision: {current['revision']}")
    render_conflict(store, r, draft, prefix, latest)
    render_live_draft(r, draft, prefix, pending, latest)
    render_editor(r, draft, prefix, pending, save, latest)
    with st.expander('Saved review history'):
        st.json(store.history(active))


def render_draft(row):
    abstained = draft_status(row) == 'Abstained'
    st.markdown('No relevant policy evidence found' if abstained else row['answer'])
    confidence = row.get('confidence') or 'none'
    color = {'high': 'green', 'medium': 'blue', 'low': 'orange', 'none': 'gray'}.get(confidence, 'gray')
    st.badge(f'Confidence: {confidence}', color=color)
    needs_evidence = row.get('needs_implementation_evidence', (row.get('evidence_assessment') or {}).get('implementation_required'))
    if not abstained and needs_evidence:
        st.badge('needs implementation evidence', color='orange')
    if row.get('citations'):
        st.subheader('Citations')
        for citation in row['citations']:
            with st.expander(f"{citation['source']} — page {citation['page']}"):
                st.text(citation['quote'])


def reset_draft(draft, base):
    draft.update(base=deepcopy(base), edit=values(base), conflict=None)
    # Run before rendering editors, or reset their values next run.
    draft['reset_widgets'] = True


def save_draft(store, r, draft, action):
    try:
        saved = store.save(r, revision=draft['base']['revision'], action=action, **draft['edit'])
    except ReviewConflict:
        draft['conflict'] = store.current(r)
        return False
    except (ValueError, OSError, sqlite3.Error) as exc:
        st.error(str(exc))
        return False
    reset_draft(draft, saved)
    return True


def render_pending(session, requested, store, r, draft, pending, save_draft):
    target = requested[-1]
    if pending:
        st.warning('Unsaved edits: save, discard, or cancel before leaving this question or changing filters.')
        if st.button('Save edit and continue'):
            if save_draft('edited'):
                session['active'] = target
                session['applied'] = requested
                st.rerun()
        if st.button('Discard changes and continue'):
            reset_draft(draft, store.current(r))
            session['active'] = target
            session['applied'] = requested
            st.rerun()
        if st.button('Cancel navigation'):
            session['reset_controls'] = True
            st.rerun()


def render_conflict(store, r, draft, prefix, latest):
    if latest['revision'] != draft['base']['revision']:
        st.warning('A newer stored revision exists. Your displayed revision and draft are retained; saving will check for a conflict.')
    if draft['conflict'] is not None:
        latest = store.current(r)
        draft['conflict'] = latest
        st.error('Review changed in another session. Your unsaved draft is retained. Compare all fields before choosing a resolution.')
        st.json({'Originally displayed': draft['base'], 'Your unsaved draft': draft['edit'], 'Current stored state': latest})
        if st.button('Keep my draft against compared revision'):
            # Only accept the revision actually compared in the previous render.
            compared = st.session_state.get(prefix + 'compared')
            if compared is not None:
                draft['base'] = deepcopy(compared)
                draft['conflict'] = None
                draft['edit']['scope_checked'] = False
                draft['reset_widgets'] = True
                st.rerun()
        if st.button('Discard my draft and load compared revision'):
            compared = st.session_state.get(prefix + 'compared')
            if compared is not None:
                reset_draft(draft, compared)
                st.rerun()
        st.session_state[prefix + 'compared'] = deepcopy(latest)
        st.caption('Keeping your draft does not save it. Review the merged draft and save explicitly; approval attestation is reset. A further concurrent change will conflict again.')


def render_live_draft(r, draft, prefix, pending, latest):
    from trustquery.review_rag import render as render_ai_draft
    with st.expander('Generate a live AI draft'):
        render_ai_draft(st, r, draft, prefix, blocked=bool(
            pending or draft['conflict'] is not None or latest['revision'] != draft['base']['revision']))


def render_sections(r):
    if st.toggle('Enable read-only section candidates', value=False):
        from trustquery.candidate_sections import locate
        hybrid = st.toggle('Use offline hybrid section ranking', value=False)
        if hybrid:
            from trustquery.hybrid_candidate_sections import locate
        if st.button('Locate full policy sections (read only)'):
            located = locate(r, enabled=True)
            st.info(located['reason'])
            for candidate in located['candidates']:
                with st.expander(f"UNVERIFIED SECTION | {candidate['source']} | page {candidate['page']} | {candidate['title']}"):
                    st.text(candidate['section_text'])
                    st.json({k: candidate[k] for k in ('score', 'overlap', 'dense_score', 'rrf_score', 'ranks', 'section_utf8_sha256') if k in candidate})
                    for ref in candidate['document_context']:
                        st.caption(f"MANDATORY CONTEXT | {ref['source']} | page {ref['page']} | PDF SHA-256 {ref['pdf_sha256']} | page UTF-8 SHA-256 {ref['page_utf8_sha256']}")
                        st.text(ref['quote'])


def render_page_context(e, prefix, index):
    label = f"Show local PDF context for {e['source']} — page {e['page']} (passage {index + 1})"
    if not st.toggle(label, key=prefix+f'evidence:{index}:context'):
        return
    try:
        evidence = page_context(e)
    except Exception:
        st.warning('Local PDF context unavailable: source missing, unsafe, unreadable, or page outside supported limits. The retrieved passage remains available; inspect the source independently.')
        return
    st.caption('CURRENT LOCAL PDF')
    if not evidence['quote_present']:
        st.warning('Retrieved passage was not found verbatim after whitespace normalization on the cited local page. Check source version, locator and context; do not assume support.')
    for number, text in evidence['pages']:
        relation = 'candidate page' if number == e['page'] else 'nearby context only'
        st.caption(f"UNVERIFIED CANDIDATE | {e['source']} | PDF page {number}/{evidence['total_pages']} | {relation}")
        st.text(text or '[No extractable text on this page]')


def render_bounded_answer(r, draft):
    from trustquery import review_gemini
    # Local presentation only: no model call, editor import or persistence.
    try:
        policy_answer = review_gemini.bounded_policy_answer(r, draft['edit']['selected'])
    except Exception:
        policy_answer = None
    if policy_answer:
        st.subheader('Bounded policy answer — literal excerpt')
        st.text(policy_answer['answer'])
        st.caption('UNVERIFIED CANDIDATE | hr_security_policy.pdf | physical page 4')
        with st.expander('Full policy source context for the bounded answer'):
            context = policy_answer['source_context']
            st.text(context['answer'])
            st.json({'references': context['references'], 'citations': []})


def render_experimental(r, draft, pending, latest, experimental_enabled):
    current = draft['base']
    if draft['edit'].get('experimental_policy'):
        st.caption('Editor source: experimental policy excerpt')
    if experimental_enabled or draft['edit'].get('experimental_policy'):
        from trustquery.review_policy import proposal
        try:
            candidate = proposal(r, enabled=True)
        except (ValueError, OSError, KeyError, IndexError) as exc:
            st.info(str(exc))
        else:
            with st.expander('Inspect experimental AI-assisted draft and source versions'):
                st.text(candidate['answer'])
                st.json({'bundle_sha256': candidate['bundle_sha256'], 'corpus_sha256': candidate['corpus_sha256']})
                for ref in candidate['references']:
                    st.caption(f"UNVERIFIED CANDIDATE | {ref['source']} | physical page {ref['page']} | PDF SHA-256 {ref['pdf_sha256']} | page SHA-256 {ref['page_sha256']}")
                    st.text(ref['quote'])
            pristine = (current['revision'] == 0 and latest['revision'] == 0
                        and current['status'] == 'unreviewed' and not current['answer']
                        and draft['edit'] == values(current) and not pending
                        and draft['conflict'] is None)
            if st.button('Use experimental draft in empty editor', disabled=not experimental_enabled or not pristine):
                draft['edit']['answer'] = candidate['answer']
                draft['edit']['experimental_policy'] = candidate
                draft['edit']['scope_checked'] = False
                draft['reset_widgets'] = True


def render_passage_selection(r, draft, prefix, reset):
    # UI-only sentinel: never send this value to the store or exporter.
    none_option = 'None — none of these passages support the answer'
    choices_key = prefix + 'passage_choices'
    if reset or choices_key not in st.session_state:
        initial = list(draft['edit']['selected'])
        # Historical empty selections have no separate intent bit in SQLite.
        if not initial and draft['base']['revision'] > 0:
            initial = [none_option]
        if not reset and 'previous_choices' in draft:
            initial = list(draft['previous_choices'])
        st.session_state[choices_key] = initial
        draft['previous_choices'] = list(initial)

    def update_passages():
        choices = list(st.session_state[choices_key])
        previous = draft.get('previous_choices', [])
        added = [choice for choice in choices if choice not in previous]
        if none_option in choices:
            # Newly added NONE wins ambiguous batch events. Adding a passage
            # to an existing NONE instead expresses a switch back to passages.
            choices = ([none_option] if none_option in added or not added
                       else [choice for choice in choices if choice != none_option])
        st.session_state[choices_key] = choices
        draft['previous_choices'] = list(choices)
        selected_ids = [choice for choice in choices if choice != none_option]
        st.session_state[prefix+'selected'] = selected_ids
        draft['edit']['selected'] = list(selected_ids)
        st.session_state[prefix+'scope_checked'] = False
        draft['edit']['scope_checked'] = False

    labels = {e['chunk_id']: f"{e['source']} — page {e['page']} (passage {index})"
              for index, e in enumerate(r['evidence_references'], 1)}
    st.multiselect('Passages to review (selection does not verify citations)',
                   [none_option] + [e['chunk_id'] for e in r['evidence_references']],
                   key=choices_key, on_change=update_passages, format_func=lambda choice: labels.get(choice, choice))


def render_approval_scope(r, draft, prefix):
    attestation = ('I reviewed every experimental source page, including the requested section and all mandatory document context, against the full question, scope, exceptions, contradictions and completeness. Approval covers only proposed synthetic policy text.' if draft['edit'].get('experimental_policy') else 'I reviewed every selected passage against the full question, including scope, exceptions, contradictions and completeness. This answer is limited to synthetic policy text.')
    st.markdown('**Approval evidence and scope**')
    selected = st.session_state[prefix+'selected']
    if st.button('Build bounded policy extract', disabled=not selected):
        st.session_state[prefix+'answer'] = policy_extract(r, selected)
    st.checkbox(attestation, key=prefix+'scope_checked')


def render_editor(r, draft, prefix, pending, save_draft, latest):
    with st.expander('Additional review tools'):
        experimental_enabled = st.toggle('Enable experimental policy drafts', value=False)
        render_sections(r)
        render_bounded_answer(r, draft)
        render_experimental(r, draft, pending, latest, experimental_enabled)
        reset = draft.pop('reset_widgets', False)
        for field in FIELDS:
            if field != 'experimental_policy' and (reset or prefix + field not in st.session_state):
                st.session_state[prefix + field] = deepcopy(draft['edit'][field])
        render_passage_selection(r, draft, prefix, reset)
        for index, reference in enumerate(r['evidence_references']):
            render_page_context(reference, prefix, index)
        render_approval_scope(r, draft, prefix)
    st.text_area('Answer draft', height=220, key=prefix+'answer')
    st.text_input('Reviewer name', key=prefix+'reviewer')
    st.text_area('Review rationale / missing evidence / rejection reason', key=prefix+'note')
    draft['edit'] = {field: deepcopy(draft['edit'].get(field) if field == 'experimental_policy' else st.session_state[prefix + field]) for field in FIELDS}
    for col, label, action in zip(st.columns(4), ['Save edit', 'Approve policy extract', 'Reject', 'Needs evidence'], ['edited', 'approved', 'rejected', 'needs_evidence']):
        if col.button(label, disabled=draft['conflict'] is not None or pending):
            if save_draft(action):
                st.rerun()
            elif draft['conflict'] is not None:
                st.rerun()
