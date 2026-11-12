# Offline questionnaire drafts

Run from `the repository root` in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m trustquery.batch data/eval/clouddesk_questionnaire.json "$env:TEMP\trustquery-drafts.json"
Get-Content "$env:TEMP\trustquery-drafts.json" -TotalCount 35
python -m pytest tests/test_batch.py -q
python -m pytest -q
```

Choose a fresh output filename on each run. The CLI refuses to overwrite any existing file and validates the entire input before opening the output. The output parent directory must exist. Errors produce exit code 2 with a message on stderr.

## Input contract

UTF-8 (optional BOM) `.json` or `.csv` files only. JSON must contain a nonempty array of objects. CSV must have unique headers and consistent row widths; ordinary blank lines are ignored by the CSV parser. Records stay in input order.

Each record requires `id`, `question`, and `response_type`. IDs and question text must be nonempty strings; surrounding whitespace is stripped. IDs are case-sensitive and must be unique after trimming. Supplied IDs are preserved across runs; missing IDs are rejected instead of assigning unstable row numbers. Supported response types are exactly `yes_no` and `short_answer`, matching the existing questionnaire.

Optional `domain` is a nonempty string, stored in output metadata. Optional `metadata` is a JSON object (a JSON-encoded cell in CSV); an empty CSV metadata cell means `{}`. Metadata is untrusted input context, never evidence. A domain cannot also be supplied inside metadata. Unknown top-level fields, including answer annotations, are rejected. Duplicate JSON keys and nonstandard numeric constants are rejected.

Minimal CSV:

```csv
id,question,response_type,domain
q1,Does your written policy require MFA?,yes_no,access_control
q2,Describe the evidence available for access reviews.,short_answer,access_control
```

Equivalent JSON:

```json
[
  {"id": "q1", "question": "Does your written policy require MFA?", "response_type": "yes_no", "domain": "access_control"},
  {"id": "q2", "question": "Describe the evidence available for access reviews.", "response_type": "short_answer", "domain": "access_control"}
]
```

## Output and boundaries

`trustquery.batch.load_questionnaire(path)` returns validated `Question` records plus source provenance. `create_drafts(path)` returns `DraftRecord` dataclasses; the CLI serializes them as a JSON array.

Every draft starts with `status: pending`, `answer: ""`, and `citations: []`. The status type reserves `needs_evidence` for future evidence processing; this loader does not assess evidence sufficiency. Provenance records the input basename, SHA-256 of its exact bytes, format, one-based logical record position, `offline_blank_draft` creation method, `not_retrieved` evidence status, `requires_human_review`, and `operational_proof: false`. It records no invented reviewer, approval, or timestamp. Identical input bytes yield deterministic records with no time-dependent values.

This module uses only the standard library. It reads only the supplied questionnaire, with no discovery of sibling files, answer-key lookup, retrieval, model loading, generation, or API calls. Policy questions and blank drafts provide no proof of deployed controls. The CloudDesk answer key remains an evaluation-only artifact and is not runtime input.

The batch foundation remains limited to validation and blank drafts. The separate retrieval adapter below adds unassessed policy passages. The separate Step 3 assessor below adds provisional lexical triage. Remaining work: semantic evidence review, bounded answer generation, citation validation, human review, and later UI/Excel support.

## Offline policy retrieval

`trustquery.batch_retrieval` connects validated input to the existing
`retrieve_and_rerank` hybrid dense/BM25, reciprocal rank fusion, and cross-encoder
APIs. It reads only the eight named CloudDesk PDFs in `data/sample`, never the
policy JSON annotations or evaluation answer key. Answer-key filenames are rejected
before reading; the questionnaire schema also rejects annotation fields. Only
question text reaches retrieval; arbitrary input metadata is retained as untrusted
context and is not used as evidence.

From the repository root, with the existing virtual environment:

```powershell
python -m trustquery.batch_retrieval data/eval/clouddesk_questionnaire.json "$env:TEMP\trustquery-passages.json" --index "$env:TEMP\trustquery-batch-index" --prepare-index --limit 3 --top-k 3
# Reuse the same index; choose a fresh output filename.
python -m trustquery.batch_retrieval data/eval/clouddesk_questionnaire.json "$env:TEMP\trustquery-passages-repeat.json" --index "$env:TEMP\trustquery-batch-index" --limit 3 --top-k 3
python -m pytest tests/test_batch.py tests/test_batch_retrieval.py -q
```

Both Hugging Face offline flags are forced before model imports. The embedding
model `all-MiniLM-L6-v2` and reranker `cross-encoder/ms-marco-MiniLM-L-6-v2` must
already be fully cached. Missing cache files fail locally; there are no Gemini
or paid calls. The first explicit `--prepare-index` embeds the corpus once into
a dedicated `batch_clouddesk_v1` collection. Later runs verify its corpus hash,
model label, document text, and locators and reuse its embeddings. No existing
collection is cleared, overwritten, or silently rebuilt. A changed corpus or
incomplete index requires a fresh index directory. Existing demo/evaluation
indexes are left alone because their corpus/model compatibility is not established.
The existing BM25 API builds its small sparse index per query; this adapter does
not change shared retrieval internals. Models are loaded once per CLI invocation.

The full questionnaire is validated before a positive `--limit` selects the first
records. IDs, order, metadata, and input hashes/record positions remain intact.
`--candidate-k` defaults to 10 and must be at least `--top-k` (default 5).
`retrieve_drafts(path, chunks=..., collection=..., reranker_model=...)` also supports
prepared resources and an injected retrieval callable for offline tests. Callers
must supply resources for the stated models and fixed corpus.

`retrieved_passages` is separate from `citations`: each passage contains its text,
PDF basename, policy stem, one-based page, zero-based chunk index, composite chunk
ID, one-based rank, and available distance/BM25/RRF/cross-encoder scores. These
scores are ranking signals, not calibrated confidence or evidence sufficiency.
Locators and text must match a supplied corpus chunk; missing or foreign locators
fail the question rather than inventing identifiers. Passage evidence is explicitly
`synthetic_policy_requirement`, `unassessed`, and `operational_proof: false`.
Policy requirements never prove implemented controls.

Answers remain blank and final citations remain empty for every outcome. Before assessment, retrieved records start `pending`; empty or failed retrieval becomes `needs_evidence`. Step 3 then applies the conservative status rule below.
Provenance records retrieval settings, corpus fingerprint, model labels, and
`retrieved_unassessed`, `empty`, or `retrieval_error`. Per-question failures retain
an explicit error and processing continues; CLI exit 1 signals such failures.
Setup/validation failures use exit 2, with no output; successful retrieval uses 0.
Output creation remains exclusive. Generation, citation validation, human review UI, and Excel support remain out of scope.

For fixed corpus, model artifacts, index and settings, the adapter preserves the
existing deterministic ranking order (including stable ties), with no timestamps
or sampling. Dependency/model upgrades and index rebuilds may change rankings;
the corpus hash and model labels are provenance, not model artifact hashes.


## Offline evidence assessment (Phase 4 Step 3)

Every retrieval record now includes `evidence_assessment`, with a question class,
per-passage classes, reasons, passage IDs, matched/missing terms, caution terms,
and explicit uncertainty. Raw passages remain `unassessed`; this separate provisional
triage does not approve them as citations. Answers stay blank and citations empty.
The question remains pending only for provisional `policy_supported`; all other
classes need evidence. Retrieval state/error provenance remains separate.

`policy_supported` means an explicit policy question has complete lexical content-term
coverage across uncautioned requirement passages. It does **not** establish semantic
entailment. `partially_supported` means some usable lexical coverage, or complete
coverage with relevant caution-bearing text. `insufficient_evidence` means no usable
policy coverage. `implementation_evidence_required` takes precedence when actual
operation is requested or policy-only intent is unclear, including empty retrieval.
`policy_coverage` and `both_policy_and_implementation_required` preserve the policy
portion of mixed requests; both policy review and operating artifacts are then needed.

This standard-library heuristic uses explicit regex cues and exact lowercase tokens.
It does not use ranking scores, NLI, answer annotations, input metadata, models or APIs.
It conservatively blocks full support on relevant negation, exceptions, TBD/unknown
or pending wording. It cannot resolve contradictions or understand paraphrases,
scope, actor, timing, quantities, cross-policy precedence, or complex demands.
Token overlap and cues can still give false positives and false negatives; compound
coverage across passages may not refer to the same control. No calibrated confidence
or human approval is implied. Inspect reasons and original passages before use.

## Evaluation-only locator measurement

Production retrieval and assessment never read the answer key. A separate module
consumes saved retrieval output and the key solely for offline diagnostics:

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
# Use an isolated copy if the prepared index must remain byte-for-byte untouched.
Copy-Item "$env:TEMP\trustquery-batch-index-20261027" "$env:TEMP\trustquery-eval-index-fresh" -Recurse
python -m trustquery.batch_retrieval data/eval/clouddesk_questionnaire.json "$env:TEMP\assessed-100-fresh.json" --index "$env:TEMP\trustquery-eval-index-fresh" --top-k 5
python -m trustquery.evaluation.batch_retrieval_eval "$env:TEMP\assessed-100-fresh.json" data/eval/clouddesk_questionnaire_answer_key.json "$env:TEMP\retrieval-report-fresh.json"
python -m pytest tests/test_batch.py tests/test_batch_retrieval.py tests/test_batch_assessment.py -q
python -m pytest -q
```

Choose fresh copy/output names. Omit `--limit` for full coverage. Evaluation reports
source-only and exact source/page hit@1/3/5 (any annotated locator, not all required
locators), eligible denominators, dataset/output counts, class distribution, errors,
empty results and all locator-miss examples. Unknown/duplicate output IDs fail.
Input hashes identify artifacts. The answer key is provisional, model-authored and
not human verified: locator recall is not answer/model accuracy or evidence validity.
Class counts are coverage diagnostics, not classification accuracy. Rows without
annotated locators are excluded from hit denominators and remain in coverage counts.
The report never feeds back into production selection or assessment.


Measured locally over all 100 questions (top-k 5, candidate-k 10, cached hybrid
retrieval and cross-encoder): source/page hit@1/3/5 = 58/83/89 out of 100;
source-only = 79/97/99 out of 100. No retrieval errors or empty results occurred.
Final lexical triage: 0 policy_supported, 5 partially_supported, 74
implementation_evidence_required, 21 insufficient_evidence. This deliberately
strict heuristic abstains heavily: boilerplate caveats and exact-token matching
prevent full support even when policy locators are found. These results do not
validate semantic classification accuracy. All 100 answers and citations stayed
empty. Eleven top-5 locator misses include cdq013 (annotated access-control p1 or
encryption p2; top result encryption p3) and cdq016 (annotated access-control p1 or
incident-response p4; top result incident-response p2). Inspect full report examples
and human-review annotations before changing retrieval or assessment rules.

## Review-ready batch processing (major deliverable 1)

Run from `the repository root`:

```powershell
.\.venv\Scripts\python.exe -m trustquery.batch_process data/eval/clouddesk_questionnaire.json "$env:TEMP\trustquery-review-results.json"
```

Use a fresh output filename. This command accepts the same validated CSV/JSON
contract described above. It defaults to local BM25 retrieval over the eight
synthetic policy PDFs, retaining only positive ranking scores. No Gemini,
neural model cache, credentials, answer key or experimental NLI is needed.
BM25 scores indicate ranking, not support or calibrated confidence.

Each output record preserves input ID/order/hash, question and metadata, retrieved
passages with PDF/page/chunk locators, conservative assessment and source-status
observations. New fields are `outcome`, `review_note`, `review_status`, `errors`,
and `evidence_references`. The latter are candidate quotes for human review and
are explicitly separate from final `citations` supporting an answer.

Outcomes:

- `implementation_evidence_required`: actual operation requested or intent unclear;
  answer and final citations stay empty. Obtain operating artifacts.
- `abstained`: no safely bounded policy answer; inspect candidate evidence.
- `processing_error`: per-question retrieval failed; error details remain and later
  questions continue.
- `policy_extract_draft`: only an exact single-requirement question whose entire
  retrieved context consists of the same literal requirement can produce a quoted
  policy extract and source/page/chunk citation. It explicitly disclaims adoption
  and implementation. This is a review draft, never an automatic approval or yes.

Every record requires human review. Metadata is never evidence. No operational
proof is inferred. The existing lexical assessor and Step3I/3J files are unchanged.
No semantic synthesis or automatic paraphrase approval is enabled.

For existing cached hybrid retrieval, add `--retrieval hybrid --index PATH` and,
for a fresh dedicated index, `--prepare-index`. Both Hugging Face offline flags
remain forced. Hybrid mode needs the cached embedding/reranking models, not Gemini.
`--top-k`, `--candidate-k` and `--limit` follow the existing retrieval contract.
Exit codes: 0 completed, 1 saved per-question errors, 2 validation/setup/output
failure. Existing output files are protected by exclusive creation.

Gemini is only needed for future optional natural-language synthesis. The existing
single-question Gemini helper is deliberately not connected: appending the top
source to generated text does not validate its claims. A future integration needs
claim-level citation validation, scope/contradiction checks and a human approval
step before answers can be used. Gemini cannot supply missing operational proof.
The Streamlit review interface and Excel export remain separate deliverables.
