# CloudDesk customer security questionnaire

SYNTHETIC / DEMO. Fictional CloudDesk Technologies, a 75-person B2B SaaS vendor.
This is a naturally phrased, model-authored draft for human review, not a completed
customer response or independently validated golden dataset.

## Files and safe use

- `clouddesk_questionnaire.json`: editable customer input, 100 questions. Each has
  `id`, `domain`, `response_type` (`yes_no` or `short_answer`) and `question` only.
- `clouddesk_questionnaire_answer_key.json`: separate evaluator-only annotations,
  exact policy excerpts, source/page/section locators, source hashes, bounded
  response examples, review needs and prohibited inferences. Do not send this file
  to retrieval, embedding, generation or customer-answer prompts.
- `tests/test_clouddesk_questionnaire.py`: offline structural, leakage, provenance,
  source-consistency and boundary regression checks, including malformed cases.

Use only the eight `data/sample/*.pdf` policy sources as the evidence corpus for
this milestone; do not recursively ingest `data/eval`, tests, or this guide.
Load only the question text into a generation request. Domain and response format
are customer-facing context, not correctness labels. Source annotations belong
only in an isolated evaluator after generation has finished.

For Yes/No operating questions, allow **Unknown / needs review**, with a short
qualification, when evidence cannot establish either answer. Policy-only evidence
must not produce an unqualified Yes or an assertion that the control is absent.
Policy questions can be answered Yes/No about the proposed written requirement.
An explicit source statement can support a bounded No: cdq031 says the supplied
policy has not been approved. This differs from an unknown operating state.
For short answers, state what the policy proposes and identify what is missing.
The bounded responses are review examples, not strings to match verbatim.

## Evidence and labels

All eight policies are treated as fictional, synthetic, proposed requirements,
not verified operating facts. Access Control v2 now has an editable JSON source,
with exact section excerpts and pointers like the other seven policies. All eight
JSON sources are checked against their PDFs (41 PDF pages in total). Page numbers
are one-based PDF pages; JSON pointers are zero-based. The duplicate
`access_control.txt` is not a ninth independent policy source.

The snapshot is a working-tree migration based on commit
`6436e85cb14d8e3304cff0a798ffbcede1693fd0`, not that commit's exact corpus.
The manifest hashes all eight PDFs and eight JSON files and is authoritative.
Any source change triggers a content and locator review.
Evidence excerpts sometimes include surrounding context to preserve limits;
each question's `expected_evidence_boundary` says what may and may not follow.
Multiple source references are expected, especially for cross-domain situations.

| Classification | Meaning | Count |
|---|---|---:|
| policy-supported | Explicit question about written policy, fully answerable as a proposed requirement | 54 |
| policy-only-not-implemented | A relevant requirement exists; adoption or operation is not evidenced, not proven absent | 15 |
| partial | Policy explains part, but a requested artifact, result, value or readiness claim is missing | 18 |
| unsupported/TBD | Requested capability, universal rule or commitment is not established or explicitly undecided | 12 |
| explicit-negative | The supplied source explicitly states a negative status; answer only within that snapshot and scope | 1 |
| potentially contradicted | Reserved for genuine conflicting source evidence in matching scope | 0 |

No genuine conflicts were identified during this model-authored review. None
were manufactured for class balance. Missing evidence, public-content encryption
scope, separate quarterly restore/annual exercise cadences, and urgent compromise
containment versus the employee termination outer limit are not contradictions.
There are 62 short-answer and 38 Yes/No prompts, across nine domains; 44 questions
cite more than one policy document. These are coverage counts, not independent
samples or measured RAG performance. Several questions share control families.

## Evaluation schema compatibility and limits

The input retains the existing `id` / `question` naming. Each answer-key record
retains `expected_answer`, `expected_source`, `expected_page`, `should_answer`.
In schema version `1.2`, fully answerable explicit policy questions use
`should_answer=true` and `response_disposition=answer_as_proposed_policy`.
An `explicit-negative` source-status answer also uses `should_answer=true`, but
with `response_disposition=answer_as_source_status`. For both, `expected_answer`
equals `bounded_response`, and the first reference is the legacy primary
source/page. There are 55 answerable projections. For other rows these expected
fields are null and
`should_answer=false`; `bounded_response` still supplies useful supported context
and an honest review outcome. This conservative mapping means “cannot fully
answer as asked,” not “suppress every policy explanation.”

The answer key is an envelope with `records` and metadata; it is intentionally
not passed directly to `load_golden_dataset`. An offline reviewer may join input
and key by ID to form familiar legacy records, but must keep the complete
multi-source references and labels. The existing binary abstention and
single-page citation metrics cannot score partial responses, unknown operational
answers or complete multi-source coverage faithfully. Do not use them to claim
accuracy for this dataset without a separately reviewed scoring design.

`scripts/run_evaluation.py` currently loads only the access-control PDF and the
older `golden_dataset.json`; it is unchanged. The old access-only unanswerable
encryption case must not be treated as a full eight-policy-corpus expectation.
No production RAG changes, threshold changes, generation runs or API calls are
part of this dataset. Tests validate data contracts and evidence consistency,
not customer control effectiveness or human semantic agreement.

## Human review gate

Every annotation is `provisional_model_authored`, `human_verified=false`,
`review_status=requires_user_review`, `approval_status=not_recorded`.
Per-record reviewer, review date, decision and notes remain null. A human must
review question realism, source scope, pages/sections, bounded response, label,
conflict judgment and evidence needed, then record actual reviewer decisions
before any gold use. Preserve synthetic provenance after review. Do not reuse
these dependent examples for both tuning and held-out accuracy claims.

Review must not convert policy requirements into deployed controls, certification,
guaranteed SLAs, committed RTO/RPO, a backup schedule, universal notification
deadlines, background checks for everyone or staffed 24/7 monitoring. The sources
do not establish any of those operating claims.

From the repository root, run with the existing environment:

```powershell
& .\.venv\Scripts\python.exe -B -m pytest tests/test_clouddesk_questionnaire.py -q -p no:cacheprovider
& .\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider
```

## Content cleanup review notes

The consolidated pass preserves all 100 IDs and the separate customer input.
cdq001 retains both access-control and personnel sources and now attributes the
review cadence and follow-up duties to their respective documents. cdq008 and
cdq041 distinguish proposed authority and escalation from observed operation.
cdq039, cdq061 and cdq090 use customer-facing wording without assuming a
recovery guarantee or referring to an internal evidence pack.

cdq031 is an explicit negative about the supplied policy's approval, supported by
“has not been approved,” rather than an inference from missing approval records.
cdq013 and cdq035 remain unknown operational answers: missing password-expiry
configuration or a supplied SOC 2 report does not prove either is absent.
cdq059 is partial: immutability/offline copies are conditional proposed options,
not a universal requirement or an implemented design. cdq079 distinguishes no
universal screening requirement from unknown completion. cdq080, cdq088,
cdq096 and cdq099 preserve the limits on signed records, employee screening,
supplier KMS architecture and tested alternatives.

The response format is Yes/No for cdq008, cdq028, cdq045, cdq053, cdq063,
cdq088, cdq090 and cdq096. Requests for artifacts or explanations still use
short_answer even when phrased as “Can you provide…”.

Validation after cleanup: **170 questionnaire checks passed**. The full suite
produced **627 passed, 13 skipped** with offline model settings and the opt-in
real-model diagnostic disabled. The first sandboxed questionnaire run produced
169 passes and one temporary-directory setup error; the normal-access rerun
passed all 170 checks. No Gemini calls were made.

The existing working fix in `tests/test_vendor_risk_management_policy.py` was
preserved byte for byte. Source PDFs and JSON files match their initial hashes;
no source policy or production code changed. Git whitespace checks include the
four untracked questionnaire files. Nothing was staged, committed or pushed.
All annotations remain model-authored and require human review; passing checks
does not establish semantic gold labels, control effectiveness or compliance.

## Access Control v2 migration

The active source is `data/sample/access_control_policy.json`; the deterministic
builder defaults to this source and the same-directory PDF. The original review
draft remains an archived draft, not another active policy. Version 2.0-proposed
retains the four exact requirements: unique accounts and administrative MFA on
page 1; quarterly access review and termination revocation within 24 hours on
page 2. Both body status blocks and synthetic footers are retained; no page moved.

The answer key is schema 1.2 / dataset version 1.2-access-control-v2. Eight records
(cdq001, cdq002, cdq005, cdq009, cdq010, cdq011, cdq013, cdq016) now cite exact
section excerpts, headings and JSON pointers instead of entire legacy pages.
Questions, IDs, labels, answerability, responses and evidence boundaries are
unchanged. Manifest hashes include the new JSON and rebuilt PDF; the former
snapshot commit is explicitly a base commit, not a claim about current bytes.

All six legacy golden labels were reviewed against the eight-policy corpus.
q001-q004 retain exact requirement support and page mappings. q005 asks about
ALL data at rest: encryption scope and the public-marketing exception (encryption
policy page 1, Scope) prevent a universal Yes or No; keep unanswerable. q006's
90-day password-expiry requirement remains unsupported. No golden label changed.
These are policy expectations, not verified implementation facts.

The frozen q002 fixture, evidence-selection seed, approved pair review and prior
MiniLM observations remain unchanged. Their regression checks compare historical
snapshots to one another and separately check the four retained active v2 facts.
Historical model outcomes are not measurements of v2. Added headings and status
text can change chunk boundaries, BM25 rankings, embeddings and NLI predictions;
rebuild any persisted index and invalidate question-only generation caches before
a future evaluation. No production evaluator or cache was changed or run here.
No new human approval, review timestamp or implementation evidence is asserted.
Earlier cleanup test counts above describe that prior run, not this migration.

### Migration validation and remaining concern

Focused validation: 227 passed, 2 skipped. Two independent builds equal the active
PDF byte for byte; the active PDF also equals the preserved review-draft PDF.
Both rendered pages were inspected and retain clear status, footers and page
numbers. The complete 16-file manifest and all 100 evidence records pass checks.
All files outside the eight edited migration files and the new active JSON were
verified byte-identical to initial inspection, including questionnaire input,
golden_dataset.json, the modified vendor test, historical fixtures/reviews and
all seven other policy sources. Git whitespace checks pass; nothing is staged.

Full offline pytest: 642 passed, 2 failed, 2 skipped. The failures are the
access-control quarterly-review parameter in test_bm25_across_seven_policies
(test_hr_security_policy.py) and test_bm25_across_eight_policies
(test_vendor_risk_management_policy.py). For "user access permissions reviewed
quarter", the active Access Control page 2 ranks fourth in the eight-policy
corpus, behind HR page 4, Backup page 3 and Encryption page 1. Both existing
tests require it in the top three. This is a measured ranking regression after
adding v2 headings/status/context, not missing requirements or incorrect pages.
The tests and production retrieval were not weakened or tuned to hide it. The
prior vendor test fix is byte-identical. Migration source/locator verification
passes, but the full regression suite is not green; Policy 2 remains untouched.
No Gemini calls, commits or pushes were made. Tests used offline model settings
with the real-model diagnostic disabled and bundled ReportLab available.


## Encryption policy wording revision

Policy 2 retains its four pages and all section locations. Shorter wording keeps
the original scope, requirements, approval boundaries, frequencies and limits.
The source and PDF remain model-authored SYNTHETIC / DEMO proposals requiring
user review, with approval_status not_recorded. The 24 Policy 2 evidence excerpts
and its two manifest hashes now match the revised JSON and reproducible PDF.
The manifest remains authoritative for current bytes; the base commit is not a
claim that the revised sources were committed. Historical snapshots are unchanged.

All 100 question IDs, questions, answerability labels and bounded responses are
unchanged. Golden q001-q004 retain their requirement support and pages. q005
remains unanswerable: scoped encryption and the public-marketing exception do
not establish encryption of all data. q006 remains unsupported; this revision
adds no password-expiry requirement. No human approval or implementation evidence
is created. Persisted indexes and generation caches still need refreshing before
any future corpus evaluation; no generation or Gemini calls were made here.

Validation for the wording revision: focused pytest 244 passed, 2 skipped;
full offline pytest 655 passed, 3 failed, 2 skipped. Repeated builds are
byte-identical, all source sections extract on their original pages, all 100
questionnaire evidence records and the 16-file manifest pass, and all four
rendered pages were visually checked. Poppler emitted missing Symbol and
ArialUnicode display-font warnings; no visible layout defect was found.

The full suite remains red on BM25 top-three expectations: the access-review
query in both HR and Vendor cross-policy tests (previously documented), plus
the HR exception query in the HR cross-policy test after this rewrite. The
requirements remain present; ranking changes are not semantic contradictions.
Retrieval code and these tests were left unchanged. No measured NLI or generation
outcomes are claimed for this revised corpus. Git whitespace checks pass.
Exactly six files changed during this revision: Policy 2 JSON/PDF, its builder
and tests, and this guide and the questionnaire answer key. Other policies,
questionnaire input, golden data and historical snapshots are byte-identical to
initial inspection. Nothing was staged, committed or pushed; no Gemini was used.


## Finalized Policy 2 and Policy 3 review draft

This section supersedes earlier validation counts for the current working tree.
Policy 2 now says Platform Engineering must maintain the certificate inventory
and must use managed cloud KMS. Storage verification and rotation-record wording
also explicitly describe required actions. Effective and next-review dates are
proposed/planned, with approval not recorded. Page 4 has shorter paragraphs and
all its original words; no requirement was removed. The builder renders those
paragraphs with consistent spacing. All four pages and 16 section locations stay.

Policy 3 is rewritten in plain language for the fictional 75-person B2B SaaS.
Its five pages and 20 section locations stay. CTO and Engineering Lead duties,
annual assessments/internal review, quarterly risk/findings follow-up, four data
classes, annual critical-vendor reassessment, onboarding/annual training,
annual and post-change continuity exercises, quarterly encrypted restores and
90-day CTO-approved exceptions reviewed monthly are retained. Detailed access,
key, incident, records, disposal and evidence limits remain. The JSON now labels
the dates proposed_effective_date and planned_next_review_date, with date_status
proposed_dates_approval_not_recorded. Its PDF metadata includes not_recorded and
footers show Page N of 5. This is ready for user semantic review, not approved.

The separate answer key refreshes 24 Policy 2 and 20 Policy 3 exact section
excerpts and four SHA256 hashes. All 100 questions, IDs, answerability labels,
bounded responses and evidence boundaries stay. Dataset version is now
1.3-policy-2-final-policy-3-wording; schema remains 1.2. The working-tree manifest
remains authoritative and the base commit is unchanged. No approval, human
verification, implementation, certification or control-effectiveness claim is
added. Both policies remain synthetic, model_authored, requires_user_review and
approval_status not_recorded.

Files changed during this continuation (relative to initial inspection):

- data/sample/encryption_key_management_policy.json
- data/sample/encryption_key_management_policy.pdf
- scripts/build_encryption_policy.py
- tests/test_encryption_policy.py
- data/sample/information_security_policy.json
- data/sample/information_security_policy.pdf
- scripts/build_information_security_policy.py
- tests/test_information_security_policy.py
- data/eval/clouddesk_questionnaire_answer_key.json
- data/eval/clouddesk_questionnaire.md
- tests/test_backup_disaster_recovery_policy.py
- tests/test_change_management_policy.py
- tests/test_vendor_risk_management_policy.py

The last three only update exact wording checks against Policy 3: required CTO
approval/acceptance, the annual/pre-change assessment phrase and capitalization
of quarterly risk review. Their obligations and ranking assertions are unchanged.
The pre-existing vendor temporary-index test fix remains intact. No policy
content other than Policies 2 and 3 changed.

Validation: focused pytest 277 passed, 2 skipped; final full offline pytest
678 passed, 2 failed, 2 skipped. Both skips are opt-in real-model diagnostics.
The two failures are the previously documented quarterly access-review BM25
query in test_hr_security_policy.py::test_bm25_across_seven_policies and
test_vendor_risk_management_policy.py::test_bm25_across_eight_policies. Their
query is "user access permissions reviewed quarter" and they require active
Access Control page 2 in the top three. Ranking tests and retrieval code were
not changed to hide the failures. The previously documented HR-exception ranking
case passes in the current corpus; no ranking tuning was done.

Source extraction verifies all nine PDF pages and sections, all 100 evidence
records and the full 16-source manifest. Two independent builds of each PDF
match its saved bytes. All nine rendered pages were visually inspected, including
Policy 2 page 4; no clipping, overlap or unreadable glyphs were found. Poppler
warned about missing Symbol and ArialUnicode display fonts; visual review found
no resulting defect. Git diff --check passes.

Policy 1 migration is preserved: active v2 JSON/PDF, its builder/test, answer-key
locators, golden data, frozen q002/evidence-selection fixtures and prior pending
test changes are unchanged by this continuation. All files outside the 13 listed
above match initial SHA256 values, and the permanent Git index is byte-identical.
Nothing was staged, committed or pushed. No Gemini calls or real-model diagnostic
runs were made. Persisted retrieval indexes and question-only generation caches
still need rebuilding/invalidation before a future corpus evaluation. Tests
confirm structural consistency and preservation checks, not human semantic
approval or evidence that controls operate.


## Policy 3 final wording and BM25 diagnosis

This section supersedes the earlier outstanding BM25 failures and validation
counts above. Policy 3 remains a synthetic, model-authored proposal requiring
user review, with approval_status not_recorded. All five pages and 20 section
locations remain. Document control now states approval not recorded once and
retains proposed effective date 2026-10-22 and planned review 2027-10-22.
The vendor opening uses two shorter sentences; due diligence, Engineering Lead
review, CTO material residual-risk acceptance before use, contractual terms,
annual/event-driven reassessment, findings and exit dependencies remain.
The records list is split into three sentences without removing any record type.
The minimum 12-month retention still applies only to KMS audit logs.

Four exact Policy 3 evidence excerpts and its JSON/PDF SHA256 hashes were
refreshed. Dataset version is 1.4-policy-3-final-wording; schema stays 1.2 and
the base commit stays unchanged. All 100 questions, IDs, original requirements,
labels, bounded responses, evidence boundaries and human-review fields remain.

The failing query was "user access permissions reviewed quarter" in both HR
and Vendor cross-policy tests. Before these edits, the HR corpus ranked
Encryption p1 (9.217072), HR p4 (8.101682), Backup p3 (6.969890), then Access
Control p2 (5.837003). Vendor ranked those same locations with scores 9.614368,
8.450205, 7.296563 and 6.050527. Retrieved chunks contain the quarterly-review
requirement. Access Control v2 is shorter and uses "reviews" rather than the
query's "reviewed"; whitespace-tokenized BM25 does not stem or strip punctuation,
so "quarter." is also distinct from "quarter". Repetition and changed document
lengths affect ranking. This is a brittle source-rank assertion, not missing
policy evidence or a semantic contradiction.

Both tests now preserve the original query and top_k=3, require the explicit
quarterly-review requirement from a relevant related policy in those results,
and independently require the exact governing requirement in Access Control
page 2's chunks. No production retriever, chunker, tokenizer or policy was tuned
for ranking. A source-only two-page BM25 index was not used as a substitute:
its different corpus statistics can yield zero IDF for discriminating terms.

Validation: focused Policy 3/HR/Vendor/questionnaire pytest 267 passed. Full
offline pytest 681 passed, 2 skipped (opt-in real-model diagnostics). Extraction
checks cover all source sections on their original pages, all 100 evidence
records and the 16-file manifest. Two independent Policy 3 builds match the saved
PDF bytes. All five rendered pages were visually checked; no clipping, overlap
or unreadable glyphs. Poppler warned about missing Symbol/ArialUnicode display
fonts without a visible defect. Git diff --check passes.

Exactly seven files changed relative to this continuation's initial inspection:
Policy 3 JSON/PDF, its tests, HR tests, Vendor tests, this guide and the answer key.
Policy 3's builder did not need changes. Other pending work is preserved;
Policies 1, 2 and 4-8 and the questionnaire input are byte-identical to initial
inspection. The permanent Git index remains byte-identical. Nothing was staged,
committed or pushed; no Gemini or opt-in model diagnostic was run. These checks
are structural validation, not human semantic approval or implementation evidence.


## Policy 4 wording review

Policy 4 is ready for human semantic review. It remains SYNTHETIC / DEMO,
model_authored, requires_user_review and approval_status not_recorded. The wording
uses shorter sentences and direct instructions; it does not claim a human author,
operating controls or approval. Proposed effective date 2026-10-22 and planned
review 2027-10-22 now use explicitly proposed metadata fields, consistent with
Policy 3. These dates are not an approved schedule.

All six pages, 24 section headings and section locations remain. Review the
following controls and caveats against the previous version:

- Page 1: scope and governing policies; CTO accountability, Engineering Lead
  coordination, assigned technical responders and personnel managers; no assumed
  retained specialists, staffed legal team, 24/7 SOC, on-call rota or response SLA.
- Page 2: proposed non-live ticket/contact route, verified alternate contacts,
  prompt reporting without proof or secrets, suspected versus confirmed incidents,
  SEV1-4 thresholds, provisional severity, reassessment and explicit handovers.
- Page 3: incident ID, facts versus hypotheses, disappearing evidence, urgent
  containment priority, SHA-256 acquisition/copy checks, custodian and transfer
  records, UTC times, clock uncertainty, read-only originals and working copies.
  Hashes alone establish neither authenticity nor admissibility. Minimize and
  encrypt evidence, restrict and record access, check authority and holds; the
  minimum 12-month retention applies to KMS audit logs, not all incident evidence.
- Page 4: urgent reversible containment without waiting for normal approvals;
  material shutdowns and residual risk remain CTO decisions. Preserve identity,
  OAuth, cloud and tenant-scope investigation, immediate key/certificate response,
  safe key disabling, historical re-encryption limits, backup decrypt dependencies,
  dual destruction approval, emergency decrypt logging/review, isolated restore
  checks, tenant separation, phased return and rollback. The 24-hour employee
  termination outer limit never delays urgent containment. No numerical RTO/RPO
  is guaranteed.
- Page 5: ransomware/extortion safety, no responder negotiation/payment authority,
  sanctions and qualified advice, no preauthorized payment or recovery guarantee,
  isolated recovery-tool validation, CTO external communication approval and
  case-specific legal/contract notification triggers and deadline calculations.
  Earlier notice may be required before an investigation ends; no universal
  deadline is invented. Retain approved messages, delivery evidence and holds.
- Page 6: closure conditions, owned uncertainties, CTO material risk acceptance,
  blameless lessons and tracked corrective actions, annual/event-driven incident
  exercises, quarterly encrypted restores, onboarding/annual awareness, controlled
  records, emergency next-business-day CTO review and encryption exception limits.
  Adoption still needs human review of staffing, contacts, authority and evidence.

Dataset version is 1.5-policy-4-wording; schema 1.2 and the snapshot base commit
are unchanged. Updated 26 exact incident excerpts across 19 records:
cdq008, cdq014, cdq015, cdq016, cdq025, cdq042-cdq053, cdq086 and cdq095.
Both Policy 4 SHA256 values in the 16-source manifest match the regenerated files.
All 100 IDs, questions, labels, answerability, bounded responses, evidence
boundaries and null human-review fields remain unchanged. The affected bounded
responses were reviewed against the revised sections; their scope remains valid.
No other policy's excerpts or hashes changed. Existing model observations remain
historical; rebuild persisted indexes and invalidate generation caches before
future evaluation. This pass did not run a corpus generation evaluation.

### HR/Vendor ranking diagnosis verified before this rewrite

The requested baseline HR/Vendor tests already passed: 60 passed, 4 skipped
(optional PDF authoring checks in the repository venv). The prior continuation
had already replaced the brittle source-rank assertions. For the unchanged query
"user access permissions reviewed quarter", actual pre-edit ranks were:

| Rank | Source / page / chunk | HR BM25 | Vendor BM25 |
|---|---|---:|---:|
| 1 | Encryption / 1 / 1 | 9.217909 | 9.615124 |
| 2 | HR / 4 / 0 | 8.102460 | 8.450903 |
| 3 | Backup / 3 / 0 | 6.970652 | 7.297248 |
| 4 | Access Control / 2 / 0 | 5.837671 | 6.051136 |

The actual first three chunks contain "user access permissions every quarter".
Access Control contains "CloudDesk reviews user access permissions every quarter."
Whitespace tokenization does not stem reviews/reviewed or strip quarter's period;
document length, repetition and corpus statistics affect scores. Requiring the
governing source itself in these broad-query top-three evidence checks was
brittle. The existing principled tests retain the query and top_k=3, require the
explicit obligation in relevant retrieved evidence, and independently check the
exact governing requirement on Access Control page 2. This verifies evidence
availability, not a guarantee of governing-source ranking. No retriever, tokenizer,
chunker, HR/Vendor test or top_k change was needed in this continuation.

After Policy 4's rewrite, the top three sources/pages/chunks remain the same.
HR scores are 7.686008, 7.000663 and 5.864658; Vendor scores are 7.949513,
7.250487 and 6.093305. Corpus wording changes affect BM25 statistics even for
other policies. All HR/Vendor retrieval checks continue to pass.

### Validation and exact scope

Focused Policy 4/questionnaire/HR/Vendor pytest: 248 passed, no skips, using
bundled ReportLab alongside the repository environment. Full offline pytest:
682 passed, 2 skipped (opt-in real-model diagnostics), no remaining failures.
PDF extraction checks every section on its original page; questionnaire checks
validate all 100 records and the complete manifest. Two independent PDF builds
match each other and the saved PDF byte for byte. All six rendered pages were
visually inspected; no clipping, overlap or unreadable text was found. Poppler
reported missing Symbol/ArialUnicode display fonts without a visible defect.
Git diff --check passes and Git reports the PDF as binary.

Exactly five repository files changed relative to this continuation's baseline:
data/sample/incident_response_plan.json, data/sample/incident_response_plan.pdf,
data/eval/clouddesk_questionnaire_answer_key.json, this review guide and
tests/test_incident_response_plan.py. The existing builder is unchanged. Added
tests check proposed date fields and section-specific incident safety boundaries.
All other captured data, scripts, tests and source files, including Policies 1-3
and 5-8, are byte-identical to baseline; the permanent Git index is byte-identical.
All pending work is preserved. No staging, commit, push, dependency changes,
Gemini calls or opt-in real-model diagnostic occurred. Structural checks do not
establish human semantic approval, implementation or compliance. Policy 4 remains
for the user's semantic review.

## Policy 5 and Policy 6 wording review

Policy 5 now splits its scope and evidence lists and explains risk-based isolation
and immutable/offline options in plain terms. Provider behavior, retention locks,
recovery access, cost and technical checks remain required; deployment is unverified.
Policy 6 has been rewritten for human semantic review, not formal approval.
Both retain six pages and all 24 section locations. The answer key version is
`1.6-policy-5-final-policy-6-wording`; exact excerpts and the 16-file manifest
pin the current sources. Review standard/normal/emergency authority, independent
review, CI and release identity, tenant tests, irreversible migrations, monitoring,
vendor/OAuth/region changes, key and backup authority, next-business-day review,
and 90-day/monthly exception limits before adoption. KMS audit logs alone have
the minimum 12-month retention rule. No human review or approval is recorded.
Rebuild persisted retrieval indexes and invalidate generation caches before reuse.


## Policy 6 final refinements and Policy 7 semantic review draft

This section supersedes earlier validation totals for the current working tree.
Policy 6 simplifies its production-change scope and splits the page 3 technical
safeguards into shorter sentences. Code, database schemas/migrations, IaC, cloud
configuration, security controls, identity, encryption, backup, recovery, vendors,
manual console changes, flags, scripts and development work affecting production
remain in scope. PR/ticket linkage, the reviewed commit and immutable artifact,
peer review, CI gates, test evidence and release blocking remain required. Its
page 6 closing sentence now reads: "This proposed policy does not provide evidence
that CloudDesk has implemented or tested these controls."

Policy 7 uses shorter, direct sentences for the fictional 75-person B2B SaaS.
All six pages, 24 section headings and original section/page mappings remain.
Review these boundaries before adoption:

- Screening is lawful and proportionate to role, access and risk. No universal
  background check, completed screening or existing HR system is asserted.
  Employee, temporary, contractor and vendor-worker responsibilities remain distinct.
- Confidentiality and acceptable-use agreements precede access. Onboarding and
  annual awareness, privileged-role preparation, unique accounts, administrative
  MFA, quarterly access reviews and approved role changes remain required.
- Employee termination access is revoked within 24 hours at the latest, preferably
  at effective departure. Urgent compromise access is revoked immediately under
  incident authority; routine approvals and evidence collection cannot delay it.
  Manager, system-owner, operator, Engineering Lead and CTO duties remain distinct.
- Offboarding includes account/session/token revocation, assets, records and service
  ownership, shared-secret exposure, verified completion, evidence holds and backup
  key dependencies. Device return does not substitute for access revocation.
- Prompt incident reporting, alternate escalation, good-faith reporting, impartial
  authorized investigation and proportionate evidence collection remain required.
  Personnel data access is restricted; lawful retention and holds require review.
  The minimum 12-month KMS audit-log rule is not a personnel retention period.
- CTO-approved exceptions expire within 90 days, are reviewed monthly and require
  fresh approval to renew. Emergency deviations need next-business-day CTO review.
  Exceptions cannot authorize unlawful screening or extend termination deadlines.

Policy 7 dates now use proposed_effective_date and planned_next_review_date with
proposed_dates_approval_not_recorded, consistent with the preceding policy drafts.
Both policies remain synthetic, model_authored, requires_user_review and
approval_status not_recorded. Policy 7 is ready for the user's semantic review;
this wording pass supplies no live SOP or completed operating evidence.

Dataset version is 1.7-policy-6-final-policy-7-wording; schema stays 1.2. Refreshed
48 Policy 6/7 evidence entries across 37 records from their exact JSON sections,
including original page/section references, and the four JSON/PDF SHA256 values.
The 100 questions, IDs, classifications, answerability, bounded responses, evidence
boundaries, human-review fields and snapshot base commit remain unchanged.
Historical model observations and frozen snapshots remain historical. Rebuild
persisted retrieval indexes and invalidate generation caches before evaluation.

### BM25 evidence and principled assertions

The requested baseline HR/Vendor tests already passed: 60 passed, 4 skipped
(optional authoring dependencies absent from the repository venv alone). Existing
fixes preserved the broad query and top_k=3, required the explicit obligation in
related-policy evidence, checked the governing Access Control page independently,
and retrieved that page with a wording-aligned query. These fixes are preserved.
Whitespace tokenization does not stem reviewed/reviews or strip punctuation;
length and corpus statistics affect ranking. No production retrieval defect was
established, and no retriever, tokenizer, chunker or top_k was changed.

For "user access permissions reviewed quarter", measured rankings were:

| Source / page / chunk | Baseline HR | Final HR | Baseline Vendor | Final Vendor |
|---|---:|---:|---:|---:|
| Encryption / 1 / 1 | 9.366513 | 9.378146 | 9.743226 | 9.755987 |
| HR / 4 / 0 | 8.249751 | 7.389491 | 8.577833 | 7.708810 |
| Backup / 3 / 0 | 6.702524 | 6.717929 | 6.994296 | 7.009131 |
| Access Control / 2 / 0 | 5.931734 | 5.933794 | 6.131913 | 6.136543 |

Encryption and HR's retrieved chunks contain the explicit quarterly obligation.
The wording-aligned governing query ranks Access Control page 2 first: final HR
16.813900 and Vendor 16.677893. This does not guarantee governing-source ranking
for arbitrary questions.

Two HR-specific assertions needed separation after the rewrite. The broad
"offboarding asset return sessions tokens holds" query ranks HR page 4 chunk 2
first (7.916687), containing revocation actions, but the device-return caveat
crosses its character boundary into chunk 3. The broad assertion now checks the
actual revocation action, and a separate caveat question must retrieve the complete
"Returning a device alone does not revoke access" sentence on HR page 4 in the
same top-three budget (chunk 3 ranks first, 21.051019).

The generic "exceptions expire monthly emergency CTO" query ranks Encryption
page 4 chunk 1 (9.513477), Information Security page 5 chunk 0 (8.218140), then
Encryption page 4 chunk 0 (8.165655). HR page 6 chunk 0 ranks fourth (8.121135).
The broad check requires the shared 90-day/monthly/CTO obligation in retrieved
evidence. A separate HR-specific unlawful-screening/termination query must retrieve
HR page 6 with the expiry, monthly review and screening restriction together.
These assertions test both general evidence availability and the specific safety
boundary without enlarging top_k or tuning policy prose for ranking.

### Final verification and scope

Focused Policy 6/HR/Vendor/questionnaire pytest: 310 passed. Full offline pytest:
745 passed, 2 skipped (opt-in real-model diagnostics). Bundled PDF authoring
libraries were made available without changing dependencies. Extraction checks
cover every section on its original page, all 100 questionnaire records and the
16-file manifest. Independent rebuilds match the saved PDF bytes. All 12 rendered
pages were visually checked with no clipping, overlap or unreadable text found.
Poppler reported missing Symbol/ArialUnicode display fonts without visible defects.
Git diff --check passes.

Exactly eight files changed relative to this continuation's captured baseline:
Policy 6 JSON/PDF, Policy 7 JSON/PDF, tests/test_change_management_policy.py,
tests/test_hr_security_policy.py, the separate answer key and this guide. Both
existing builders are unchanged. Other captured data, scripts, tests and source
files, including Policy 8, the questionnaire input and historical snapshots, are
byte-identical to baseline. The permanent Git index is byte-identical. Existing
Policy 1-6 and dataset work is preserved. Nothing was staged, committed or pushed;
no Gemini call or real-model diagnostic was run. Passing checks are structural
validation, not human semantic approval, compliance or implementation evidence.


## Policy 7 final refinements and Policy 8 semantic review draft

Dataset version is `1.8-policy-7-final-policy-8-wording`; schema 1.2 and the
snapshot base commit remain unchanged. Policy 7 clarifies personnel coordination,
splits onboarding approvals from provisioning, and groups proposed personnel
records separately from access/security evidence. All original evidence categories
remain. Policy 8 now uses shorter, direct business prose. Both retain six pages,
24 section headings and their original page/section locations.

Policy 8 remains for the user's semantic content review. Check vendor inventory,
data flows, tier criteria and separately flagged criticality; business/system
ownership, Engineering Lead oversight and CTO material-risk approval; pre-use
assessment, annual critical-vendor reassessment and reviews after changes/incidents.
Confirm security/privacy evidence limits, tenant access, cloud shared responsibility,
subprocessors/fourth parties and concentration exposure. Certifications are neither
assumed nor universally required. Vendor workers need not share CloudDesk employee
screening rules, and vendor systems need not copy its KMS architecture.

Review contract confidentiality, permitted use, incident cooperation and notification
case by case. No universal notification deadline or geographic restriction is added.
Preserve access termination, verified data return/deletion, retained backups/holds,
key dependencies, quarterly encrypted restores, annual/event-driven continuity
exercises, portability and exit planning. Recovery targets remain TBD. Planned
exceptions need CTO approval, expire within 90 days and receive monthly review;
emergency deviations need next-business-day CTO review without delaying containment.

Policy 8 date metadata is explicitly proposed, consistent with the other drafts.
Both policies remain SYNTHETIC / DEMO, model_authored, requires_user_review and
approval_status not_recorded. No deployed assurance process, completed assessment,
actual personnel record, certification or human approval is asserted.

Refreshed 58 Policy 7/8 evidence entries directly from their JSON section pointers
and the four JSON/PDF SHA256 values. All 100 question IDs and customer questions
remain unchanged and separate from the answer key. Labels, bounded responses,
answerability and null human-review fields are preserved. Historical snapshots
remain historical; rebuild persisted indexes and invalidate generation caches
before future evaluation. No generation evaluation or Gemini API call was run.

### Current validation and exact scope

Focused HR/Vendor/questionnaire tests: 284 passed. Full offline suite: 771 passed,
2 skipped (opt-in real-model diagnostics). Both PDFs retain six searchable pages;
all sections extract on their original pages. Independent rebuilds are byte-identical
to the saved PDFs. All 12 rendered pages were visually inspected with no clipping,
overlap or unreadable glyphs. Poppler reported missing Symbol/ArialUnicode display
fonts without visible defects. Existing builders are unchanged.

The broad query "user access permissions reviewed quarter" ranks Encryption p1,
HR p4, Backup p3 and Access Control p2 in both corpora. Final HR scores are
9.382904, 7.394463, 6.722943 and 5.936200; Vendor scores are 9.775736, 7.729941,
7.031508 and 6.144462. The first two results contain the explicit quarterly
obligation. The wording-aligned query "CloudDesk reviews user access permissions
every quarter." ranks governing Access Control p2 first (HR 16.763411; Vendor
16.564837). Whitespace BM25 distinguishes reviewed/reviews and quarter/quarter.;
corpus lengths and term frequencies also affect ranks. Existing broad-query,
specific-governing-query and corpus-preservation checks pass with top_k=3.
No production retriever, tokenizer, chunker or retrieval-budget change was needed.

Exactly eight repository files changed relative to this continuation's baseline:
data/sample/hr_security_policy.json, data/sample/hr_security_policy.pdf,
data/sample/vendor_risk_management_policy.json, data/sample/vendor_risk_management_policy.pdf,
data/eval/clouddesk_questionnaire_answer_key.json, this guide,
tests/test_hr_security_policy.py and tests/test_vendor_risk_management_policy.py.
All other captured data, scripts, tests and source files, including the customer
questionnaire input and historical snapshots, remain byte-identical. The permanent
Git index is byte-identical. Git diff --check passes. No staging of the permanent
index, commit, push, dependency installation, Gemini call or model diagnostic ran.
There are no outstanding structural or retrieval test failures. Policy 8 still
requires the user's semantic review; passing checks do not record approval.

## Policy 8 content refinements and eight-policy integrity audit (2026-10-27)

This section supersedes earlier validation counts and "current" wording notes
above. Dataset version: `1.9-policy-8-content-review`; schema remains `1.2`.
The base commit remains historical; the 16-file SHA256 manifest identifies the
actual working-tree sources. This is a model-authored review, pending human
semantic review. It records no human sign-off or operating evidence.

Policy 8 page 1 now describes the proposed requirements in natural external
policy language and explicitly says controls are not established as operating.
The page 3 assurance section uses shorter sentences while retaining questionnaires,
architecture explanations, relevant policies, independent summaries if available,
scoped validation and recovery/deletion descriptions. Relevance, dates, scope,
service coverage, exclusions, provenance, remediation status, unavailable evidence,
unverified claims, certification limits and restrictions on confidential reports,
plaintext secrets and unnecessary personal data remain. Page 6 now says the
policy does not establish completed vendor assessments, implemented controls or
independent certification. All substantive requirements, six pages and 24 section
locations remain. Provenance remains model_authored; synthetic=true,
requires_user_review and approval_status=not_recorded remain.

Exact excerpts changed in cdq035 and cdq093. Other excerpt pointers and sections
are unchanged. Both Policy 8 source hashes were refreshed; all 16 manifest entries
were verified against the actual source files. No historical snapshot was rewritten.

### Cross-document obligations and scope review

All eight complete JSON sources were reviewed and their PDF agreement checked.
There are 41 searchable PDF pages. No substantive contradiction requiring another
policy change was identified in this pass. References below are one-based PDF pages.

| Obligation | Governing source and cross-document checks | Review conclusion |
|---|---|---|
| Unique accounts and administrative MFA | Access Control p1; Encryption p1/p3; Information Security p3; Incident p4; Backup p3; Change p3; HR p3; Vendor p4 | Administrative MFA is retained; no universal customer MFA, SAML or SCIM capability is inferred. |
| Quarterly access review | Access Control p2; Encryption p1/p3; Information Security p3/p5; Incident p4; Backup p3; Change p3; HR p4; Vendor p4 | Quarterly user/grant reviews remain. Event-driven changes and offboarding do not wait for the quarterly cycle. |
| Employee termination within 24 hours | Access Control p2 and corresponding access references above; HR p4; Vendor p6 | This is an outer employee limit, not permission to delay urgent containment or a universal vendor waiting period. |
| TLS, AES and KMS | Encryption p1-p4; Information Security p2/p3; Incident p3/p4; Backup p3; Change p3/p5; HR p5; Vendor p4 | TLS 1.2/1.3 and AES-256 retain their information/storage/backup scope. Public marketing storage exception does not exempt administrative/authentication traffic. Vendor assessment does not impose CloudDesk's own KMS architecture universally. |
| Key authority and recovery | Encryption p3/p4; Incident p4; Backup p3/p4; Change p3/p5; HR p4/p6; Vendor p4/p6 | No default decrypt rights for key administrators; approved grants, restricted old versions, retention/hold checks and dual destruction approval remain. Exit/termination alone does not authorize key destruction. |
| Quarterly restore versus annual continuity | Encryption p4; Information Security p4; Incident p6; Backup p5; Change p3; Vendor p5 | Representative encrypted restore at least quarterly and continuity exercises at least annually/after material change are distinct. Neither proves universal recoverability or substitutes for the other. |
| Critical vendor reassessment | Information Security p3; Backup p5; Change p5; HR p2; Vendor p2/p4 | Annual critical-vendor reassessment plus change/incident triggers remain. Criticality is tracked separately from risk tier. No completed assessment or certification is inferred. |
| Bounded exceptions | Encryption p4; Information Security p5; Incident p6; Backup p6; Change p6; HR p6; Vendor p6 | Maximum 90 days, monthly review, fresh renewal assessment/approval and next-business-day emergency CTO review remain. No extension of termination limits or destructive authority. |
| Approval versus evidence | Status/adoption sections in all eight; Information Security p5; Change p2/p5; Vendor p3/p6 | Proposed dates, approver roles, purchasing approval, merged PRs and passing tests do not establish adoption, successful operation or independent assurance. |

RTO/RPO, backup frequency and retention remain TBD pending BIA/business/contractual
review. Minimum 12-month KMS audit-log retention is not universal record retention.
Notification duties remain case-specific, with earlier notice when required.
Screening is role/risk/jurisdiction dependent; no blanket background-check rule,
24/7 staffing, guaranteed SLA, certification or actual control results were added.

### Review of all 100 classifications and input separation

Reviewed cdq001-cdq100 against their questions, bounded responses and source scope.
Retained 54 policy-supported, 15 policy-only-not-implemented, 18 partial,
12 unsupported/TBD and one explicit-negative; zero genuine conflicts identified.
No label, ID, question, answerability or human-review field changed.
Policy-supported means the proposed written requirement answers the policy
question. It does not mean implemented. The 15 operating questions remain
unverified, not proven negative. Partial rows combine supported policy context
with missing artifacts/results/values/readiness. Unsupported/TBD rows do not
establish the requested capability or universal commitment. cdq031 alone supports
a bounded negative about this supplied policy's explicitly unapproved status.

The 100-question customer input is byte-identical to the captured starting state.
It contains only id, question, domain and response_type; annotations and evidence
remain in the separate evaluator key. Existing tests reject leaked fields and
unsafe legacy projections. A future pipeline must enforce the documented source
allowlist; file separation alone cannot prevent a future indiscriminate ingester.
No generation run or operational answer evaluation was performed here.

### HR/Vendor BM25 diagnosis

The two previously reported failures already had meaningful fixes in the pending
working tree. Inspected the actual whitespace tokenizer, extracted chunks and
scores and reran both corpora. No further retrieval change was needed.
For `user access permissions reviewed quarter`, the first four results are:

| Corpus | Encryption p1 | HR p4 | Backup p3 | Access Control p2 |
|---|---:|---:|---:|---:|
| Seven-policy HR | 9.382904 | 7.394463 | 6.722943 | 5.936200 |
| Eight-policy Vendor | 9.779561 | 7.733824 | 7.035182 | 6.146808 |

The first two chunks contain explicit quarterly user-access-review wording.
The broad query does not rank the governing policy in the first three because
whitespace BM25 distinguishes reviewed/reviews and quarter/quarter-with-a-period,
and corpus length/term frequencies also affect scores. The existing assertions
preserve the broad query and top_k=3 and require explicit equivalent evidence.
They separately query `CloudDesk reviews user access permissions every quarter.`
and require Access Control p2, which ranks first (HR 16.763411; Vendor 16.570548).
The authoritative corpus requirement is independently checked. Inspecting four
ranks was diagnostic only; the test retrieval budget remains three. No tokenizer,
chunker, production retriever or evaluator was changed or weakened.

### Validation and scope

Final focused HR/Vendor/questionnaire validation: **285 passed, 0 failed, 0 skipped**.
Full offline pytest: **772 passed, 0 failed, 2 skipped**. The two skips are opt-in
real-model diagnostics, disabled deliberately. Bundled ReportLab/Pillow were
made available without installing dependencies. An initial focused run without
the optional authoring path had 281 passed and 4 authoring skips; the final run
executes those rebuild checks too.

Two independent Policy 8 builds equal the saved PDF byte for byte. Its six pages
retain every JSON section on its original page. All six rendered pages were
visually inspected: legible text, consistent headings/footers, no clipping or
overlap. Poppler reported missing Symbol/ArialUnicode display fonts; no visible
defect was found. Manifest, exact evidence pointers and complete eight-source
JSON/PDF agreement pass. Git diff --check passes, including whitespace validation
of the existing untracked questionnaire files. The permanent index is unchanged.

Exactly five repository files changed relative to this turn's captured baseline:
Policy 8 JSON/PDF, its test file, the questionnaire answer key and this guide.
Every other captured tracked/untracked file is byte-identical, including the
other seven policy sources, HR tests, customer input and historical fixtures.
Existing pending changes remain. Nothing was staged, committed or pushed.
No Gemini, other paid API, real-model diagnostic or batch UI work was performed.

### Recommendation and next steps

Proceed to human content review before gold use or customer answers. Review all
100 labels/responses, source boundaries, realistic wording and evidence requests;
record actual reviewer decisions separately while preserving synthetic provenance.
Validate feasible owners, substitutes, contacts, approval gates and contractual
constraints before adopting any policy. Structural checks are not human sign-off.

After sign-off, design an offline batch-answering path around the eight-PDF
allowlist, stable IDs, source-manifest fingerprints and isolated evaluator data.
Rebuild persisted indexes and invalidate generation caches when source hashes
change. Require page/section citations, bounded policy language, Unknown/needs
review for operating questions, and per-row failures/retries without losing IDs.
Review scoring for partial answers, multi-source coverage and operational unknowns
before reporting accuracy; existing binary/single-page metrics are insufficient.

Then design Excel export with separate customer-response and internal-review
sheets: stable ID, question, bounded answer, disposition, citations and review
status; keep evaluator labels/excerpts out of the customer-facing sheet. Preserve
provenance and source hashes in internal metadata, use actual review decisions,
escape formula-leading user text, and validate ID/count round trips. This is a
roadmap only; no batch UI or Excel exporter was implemented.
