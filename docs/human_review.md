# Human review interface (Major Deliverable 2)

Run in Windows PowerShell from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[review]"
.\.venv\Scripts\python.exe -m streamlit run src/trustquery/review_app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
.\.venv\Scripts\python.exe scripts/test_release_offline.py -k review
```

Open the loopback URL Streamlit prints. The app starts with no selected batch.
Upload a questionnaire CSV/JSON and process it offline, or reopen an existing
trusted results JSON and its paired SQLite database in the sidebar. See
[upload schema and limits](questionnaire_upload.md). Processing uses BM25 over
eight hash-checked bundled synthetic proposed policies; arbitrary policy PDF
upload is unsupported. Gemini, external APIs and model inference are disabled
in this upload workflow. Existing experimental tools remain separately opt-in.

Search by ID/question, filter domain, batch outcome or saved review status, and
navigate with the question selector or Previous/Next. Expand candidates to read
PDF basename, one-based page, zero-based chunk index, chunk ID and full quote.
These are candidate references, never verified citations. Assessments, recorded
errors and provenance are visible separately. The policies are synthetic and do
not establish approval, adoption, implemented controls or compliance.

Edit the draft and enter a reviewer name. Each question retains its originally
shown saved revision and all draft fields across reruns in that browser session.
Changing questions or filters with unsaved edits keeps the original editor open
and requires Save edit and continue, Discard changes and continue, or Cancel
navigation. Save failures keep the draft and pending navigation; cancel restores
the prior filters and question. Discard explicitly reloads stored state. Batch and
database path changes are blocked while drafts are dirty, including changes
submitted in the same request as an edit. Unsaved drafts are session memory only:
closing/disconnecting the browser session or restarting the server may lose them.
Save explicitly for durable storage. Save edit invalidates any prior approval. Reject and
Needs evidence require a reason. Approval requires selected candidate passages,
a recorded scope/completeness/contradiction review and rationale, plus the exact
bounded text produced by Build bounded policy extract. Approval records a human
review of a synthetic literal extract only; it never marks candidate evidence as
verified, populates final citations, or records operational proof. Operational
questions, failed retrieval, arbitrary paraphrases and unsupported claims cannot
be approved. Human review quality cannot be established by a checkbox; reviewers
must inspect the question and full source context before attesting.

State lives in `data/reviews/major_deliverable_2.sqlite3`, separate from input.
Use a different `.sqlite3` path for an independent review. Exact batch-byte SHA-256
binding rejects reuse with changed input. SQLite transactions persist across
reruns/restarts; append-only application events retain all revisions, reviewer,
UTC database timestamp, rationale, selections and draft. Concurrent stale saves
fail without changing stored history. A conflict displays the original saved
state, your unsaved fields, and the current stored state. Save actions are blocked
until you deliberately keep your draft against the compared revision or discard
it and load that compared revision. Keeping a draft clears the scope attestation
and requires a separate explicit save. Resolution uses the revision actually
shown in the preceding comparison, so another concurrent save still conflicts.
No automatic merge or overwrite occurs. Each browser session must resolve its
own conflicts; reviewers should check every field in the comparison. Back up the SQLite database while the
app is stopped. This local tool has no authentication, signature verification,
external implementation-evidence ingestion, semantic claim validator, graphical PDF renderer,
tamper-resistant audit service. Keep it bound to localhost.
No production runtime or evaluation label is updated, including CDQ021. Input
citations remain in the original immutable batch, and are not promoted by review.
The app's review database is its durable output. Export the latest saved state with the separate CLI described in [Excel export](excel_export.md).

## Local evidence context

Expand a candidate and enable **Show local PDF page text and nearby context**.
The viewer reads only the eight named policy PDFs in repository `data/sample`,
independent of the batch path or working directory. Candidate paths, URLs,
absolute paths, traversal, alternate streams and redirected files/directories
are not accepted. Missing, malformed, encrypted, oversized (>20 MiB / 200 pages)
or out-of-range sources show an unavailable warning without losing drafts.

The batch passage and current local PDF text are labeled separately, with
one-based physical PDF pages and zero-based chunk indices. Previous and next
pages are context only; they do not join the selected evidence. A whitespace
normalized passage mismatch warns about possible version/locator differences.
A match is not evidence verification or a source-version integrity check.
There is no automatic retrieval-time PDF hash binding. Context is read fresh
on demand and is not cached across questions/batches. The viewer displays plain
text, never embeds active PDF content, and does not preserve graphical layout,
images or tables; each page is capped at 100,000 characters. Inspect the original
PDF independently when visual structure matters. All references remain explicitly
UNVERIFIED CANDIDATE, with no claim of policy implementation. Viewer interactions
do not save reviews, reset approval attestation, or bypass navigation/conflict guards.


### Empty evidence and returning after refresh

Use **None — none of these passages support the answer** in the passage multiselect when candidates do not
support the answer, then record a rationale and choose Reject or Needs evidence
(or Save edit). NONE clears selections and scope attestation; it is stored as an
empty list and creates no verified citations. Unsupported approval remains blocked.

After page refresh, expand **Reopen existing review run**, select your run UUID
under `data/reviews/ui_v1`, and click **Reopen selected run**. Saved SQLite decisions,
edits and revision checks survive; the file uploader and unsaved session drafts do
not. Refresh never processes a new run automatically. Save or discard drafts before
switching runs. See questionnaire_upload.md for validation and export limits.


Optional Gemini preview (v1 experiment)

Offline processing remains the default. In the repo-local `.env`, set `GEMINI_API_KEY=<your authorized key>` and `TRUSTQUERY_GEMINI_MODEL=gemini-3.5-flash-lite`, then restart Streamlit from the repo directory. Do not share or commit `.env` or credentials. This path uses the existing google-genai dependency but a separate single-attempt models client; it does not use the legacy interactions retry loop. No API calls were used to validate this feature. The optional preview defaults to `gemini-3.5-flash-lite`; an explicit `TRUSTQUERY_GEMINI_MODEL` overrides it (an empty value disables configuration). Verified against the [official Google model documentation](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite) on 2026-11-09. Account access and live behavior remain unverified. Existing process environment values take precedence over `.env`; restart Streamlit after changing settings. To revert explicitly, set `TRUSTQUERY_GEMINI_MODEL=gemini-3.6-flash`. The separate legacy interactions client is unchanged. The existing 60-second default timeout, single attempt with no automatic retries, per-question consent and evidence/export checks are unchanged.

Open one question, select supporting candidate passages, expand Optional Gemini draft preview, enable the toggle, inspect the outgoing synthetic context and authorize its transfer to Google, then click Generate draft with Gemini. NONE refuses generation. Only finite inspected policy-only questions pass; operational proof, compliance, approval, SLA, unknown or ambiguous scope refuses before a call. Hash changes, oversized context and unbound selected passages also refuse. The full inspected extraction/context is used rather than relevance ranking as proof.

The response must exactly reproduce the bounded extraction and reference structure. Unsupported paraphrases, additional claims, malformed/truncated JSON and altered citations are discarded. This verifies structure and byte provenance only, not semantic truth or independent contradiction adjudication. Output stays experimental/unverified in a session-only preview, never overwrites the editor, saves, approves or creates verified citations, and is excluded from Excel. Human review and existing candidate/verified citation separation continue unchanged. Refresh loses previews.

Requests disclose the question and at most 24 KB of synthetic context to Google; editor text, notes, reviewer identity and history are excluded. Provider processing/retention and charges apply. Maximum 8192 output tokens, 20-second request timeout, no retries, 60-second cooldown and five attempts per browser session (failed calls count). Refresh/new sessions reset this local throttle; it is not a global spend cap. Configure provider billing/quota controls. Quota/429, network and provider errors display a sanitized failure; retry only by another explicit click after cooldown. No arbitrary PDF uploads or batch generation.

Dependency vulnerabilities, licensing and publication gates remain unresolved. This feature does not establish release/publication readiness. Native Excel and real Gemini behavior remain unverified.
