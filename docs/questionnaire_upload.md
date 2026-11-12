# Questionnaire upload in v1

From a Python 3.12 source checkout with the review dependencies installed, run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run src/trustquery/review_app.py --server.address 127.0.0.1 --server.maxUploadSize 1 --browser.gatherUsageStats false
```

Open the printed loopback URL. This local UI has no authentication; protect its
files and keep it bound to loopback. Start a questionnaire review accepts one
UTF-8 CSV or JSON questionnaire. Download the CSV template, select a file, then
choose Process questionnaire offline. The app starts empty and does not open
historical batches automatically.

## Schema and limits

CSV has a unique header row; JSON is a nonempty array of objects. Required fields
are `id`, `question`, `response_type`; allowed optional fields are `domain` and
`metadata`. IDs and questions must be nonempty strings; IDs are trimmed, unique
and case-sensitive. `response_type` is exactly `yes_no` or `short_answer`.
`domain`, when supplied, is a nonempty string. `metadata` is an object (a JSON
object string in CSV); its optional `domain` must be a string. Do not supply
domain both at top level and in metadata. Unknown top-level fields, duplicate
JSON keys/CSV headers/IDs, malformed rows, invalid UTF-8, nonfinite numbers,
unpaired surrogates and XML-incompatible controls are rejected. UTF-8 BOM is accepted.

Application limits: 1 byte through 1 MiB, 1 through 100 questions, 128 characters
per ID, 10,000 characters per question and 10,000 in serialized metadata. Plain
filenames only, at most 128 characters, `.csv`/`.json` extensions (case-insensitive);
paths and names containing `answer_key` are rejected. Content is parsed against
the schema; an extension/MIME label alone is insufficient. Server upload size
should also be limited with the launch flag above; application validation runs
after Streamlit receives bytes and is not a transport-level resource guarantee.

```csv
id,question,response_type
q1,What does the written access control policy require for access reviews?,short_answer
```

```json
[{"id":"q1","question":"What does the written access control policy require for access reviews?","response_type":"short_answer","domain":"Access"}]
```

Question text is data, never executable instructions. Formula-like values remain
literal XLSX strings; they do not become formulas, hyperlinks, macros or DDE.

## Sources, processing and review

Policy PDF upload is **not supported**. Only the eight named bundled CloudDesk
synthetic proposed-policy PDFs are used. SHA-256 is checked against the bundled
manifest before and after ingestion; a mismatch stops processing. Input bytes,
corpus/chunk binding and eight PDF hashes are recorded in result provenance.
Hashes establish version binding, not source trust, adoption or implementation.
These fictional, model-authored policies provide no compliance evidence.

Upload processing uses local positive-score BM25 candidates. No Chroma index,
embeddings, reranker, NLI, Gemini/HF calls, model inference or model downloads are
used. Existing API/LLM configuration is preserved. Default processing is
conservative: candidate retrieval is not support, and most questions abstain;
operational questions require implementation evidence. The shared processor can
emit only a narrowly bounded literal policy draft, still requiring human review.
Existing experimental policy tools are separately opt-in and unapproved.

Each successful process allocates a new UUID directory under
`data/reviews/ui_v1/`, containing `results.json` and `review.sqlite3`. Repeated
uploads create separate runs and do not overwrite prior reviews. No automatic
retention cleanup runs. After refresh, expand **Reopen existing review run**,
choose the UUID shown in the Saved run path, and click **Reopen selected run**.
No re-upload is needed. The selector lists only immediate UUID directories under
`data/reviews/ui_v1`; opening requires existing paired files and a matching stored
batch hash. Missing, redirected or mismatched files fail without creating a run.
Advanced trusted sidebar paths remain available. The SQLite database binds to the exact result bytes. It is local mutable
storage, not an externally immutable audit service or multi-user security boundary.

Inspect candidates, edit, select passages and record an explicit human decision.
**None — none of these passages support the answer** in the passage multiselect clears all selected passages
and resets scope attestation. Save edit, Reject and Needs evidence accept an empty
selection; Reject/Needs evidence require a reason. An empty selection cannot build
or approve a bounded candidate policy extract. NONE is stored as `selected=[]`,
not a fabricated candidate or citation. Existing rejected reviews remain intact.
Nothing is automatically saved or approved. Approval applies only to the exact
bounded synthetic policy extract after scope attestation and rationale; it never
establishes implemented controls. Reject/Needs evidence require reasons. Stale
saves conflict rather than overwrite; compare and resolve explicitly. Dirty drafts
block run changes and exports. Unsaved drafts exist only in the current session.

Prepare reviewed Excel exports saved decisions from one read-only SQLite snapshot.
The workbook separates reviewed answers and approved policy extracts from
UNVERIFIED candidates. Verified citations stay empty, including after approval.
Large or XML-incompatible edited cells fail export rather than truncate content.
Protect exports: they contain questions, full text, reviewer identities and notes.

## Verification limits

Step 6 AppTest exercises processing/review/export widgets with a mocked file
selection because Streamlit AppTest has no uploader input setter. The real
adapter is also tested directly with temporary files/databases, and openpyxl
independently reads generated XLSX bytes. These checks do not verify browser
file selection, HTTP upload delivery, download delivery or native Excel layout.
Computer Use observed the loopback address/document URL, but its upload action
returned no readable follow-up state; no bypass or alternate browser automation
was attempted. Browser upload/download verification remains blocked/unverified.
See [Step 6 report](../release/STEP6_REPORT.md) for exact outcomes and repro commands.


Optional Gemini preview (v1 experiment)

Offline processing remains the default. In the repo-local `.env`, set `GEMINI_API_KEY=<your authorized key>` and `TRUSTQUERY_GEMINI_MODEL=gemini-3.5-flash-lite`, then restart Streamlit from the repo directory. Do not share or commit `.env` or credentials. This path uses the existing google-genai dependency but a separate single-attempt models client; it does not use the legacy interactions retry loop. No API calls were used to validate this feature. The optional preview defaults to `gemini-3.5-flash-lite`; an explicit `TRUSTQUERY_GEMINI_MODEL` overrides it (an empty value disables configuration). Verified against the [official Google model documentation](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite) on 2026-11-09. Account access and live behavior remain unverified. Existing process environment values take precedence over `.env`; restart Streamlit after changing settings. To revert explicitly, set `TRUSTQUERY_GEMINI_MODEL=gemini-3.6-flash`. The separate legacy interactions client is unchanged. The existing 60-second default timeout, single attempt with no automatic retries, per-question consent and evidence/export checks are unchanged.

Open one question, select supporting candidate passages, expand Optional Gemini draft preview, enable the toggle, inspect the outgoing synthetic context and authorize its transfer to Google, then click Generate draft with Gemini. NONE refuses generation. Only finite inspected policy-only questions pass; operational proof, compliance, approval, SLA, unknown or ambiguous scope refuses before a call. Hash changes, oversized context and unbound selected passages also refuse. The full inspected extraction/context is used rather than relevance ranking as proof.

The response must exactly reproduce the bounded extraction and reference structure. Unsupported paraphrases, additional claims, malformed/truncated JSON and altered citations are discarded. This verifies structure and byte provenance only, not semantic truth or independent contradiction adjudication. Output stays experimental/unverified in a session-only preview, never overwrites the editor, saves, approves or creates verified citations, and is excluded from Excel. Human review and existing candidate/verified citation separation continue unchanged. Refresh loses previews.

Requests disclose the question and at most 24 KB of synthetic context to Google; editor text, notes, reviewer identity and history are excluded. Provider processing/retention and charges apply. Maximum 8192 output tokens, 20-second request timeout, no retries, 60-second cooldown and five attempts per browser session (failed calls count). Refresh/new sessions reset this local throttle; it is not a global spend cap. Configure provider billing/quota controls. Quota/429, network and provider errors display a sanitized failure; retry only by another explicit click after cooldown. No arbitrary PDF uploads or batch generation.

Dependency vulnerabilities, licensing and publication gates remain unresolved. This feature does not establish release/publication readiness. Native Excel and real Gemini behavior remain unverified.


Exact cdq001 wording is now supported by the Gemini-only preview: "How often does your written policy call for staff access reviews, and who is meant to follow up on unnecessary access?". It binds the complete HR section 4 (Transfers, leave and offboarding) and all six HR pages to the existing eight-PDF corpus hashes. Quarterly reviews and Engineering Lead follow-up are proposed synthetic requirements, not completed operations. All existing opt-in, NONE, response validation and no-save/no-approval guards remain. Other question variants still refuse; the general review-policy approval matcher is unchanged.
