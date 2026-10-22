# Evidence selection seed (Step 3, first substep)

`evidence_selection_seed.json` is a provisional, offline fixture at base commit
`90fdc1e`, not a gold dataset. It contains 6 candidate pools / 14 evidence pairs:
4 entailments, 7 neutrals, 3 contradictions; 11 local page snapshots and 3
explicitly synthetic passages. Two pools contain both support and contradiction.
There are 5 distinct claims; the accounts claim is reused for the contradiction-with-neutral
diagnostic. These are dependent examples, not independent samples.

| Pool | Expected relationships | Conflict in supplied pool |
|---|---|---|
| q002-baseline | entailment, neutral | no |
| mfa-conflict | entailment, neutral, synthetic contradiction | yes |
| review-conflict | neutral, entailment, synthetic contradiction | yes |
| revocation-supported | neutral, entailment | no |
| encryption-unsupported | neutral, neutral | no |
| accounts-contradiction-with-neutral | neutral, synthetic contradiction | no |

## Sources and annotation contract

Local text is copied exactly from `tests/q002_nli_fixture.json`, whose two pages
are checked against `load_pdf(data/sample/access_control_policy.pdf)` in tests.
Questions come from `data/eval/golden_dataset.json`; claims are fixed in the JSON.
The q002 claim is copied verbatim; other claims are model-authored assertions
based on local policy or the encryption question. `data/sample/access_control.txt`
duplicates the quarterly review assertion and is deliberately not another example.
These are repository sample documents, not independently verified real policies.

All labels and conflict annotations are **provisional_model_authored**, with
`human_verified: false` and `requires_user_review: true`. The old q002 fixture's
`human_expected_label` naming does not establish independently verified human
review for this dataset. No model predictions or reranker scores define labels.

Each evidence record has text, source, one-based page, expected relationship,
relevance, rationale and provenance. Synthetic sources and text begin with
`SYNTHETIC`; their page 1 is a virtual locator. Synthetic passages are authored
adaptations of the explicit opposing policies explored in Step 2, not quotations
from a local document. Step 2's report and results were inspected outside the repo;
this JSON is self-contained and does not require those files to run validation.

`relevant_to_claim` means direct support or direct refutation of the precise
claim. Neutral pages here are irrelevant even when they share access-control
vocabulary. This seed has no relevant-but-neutral example; the equivalence between
relevance and non-neutral labels is a seed convention, not a general NLI rule.
NLI direction is evidence (premise) to claim (hypothesis).

`genuine_conflict_exists` is a provisional **pool-level** E+C annotation under
the scenario assumptions, repeated contextually through its enclosing case.
Synthetic conflicts assume the same organization, subject population, scope and
effective time, with no precedence resolving them. The field does not assert a
real-world policy conflict. Contradiction alone is not a conflict. In the final
pool, supporting page 1 is intentionally absent; in the encryption pool neither
page establishes or denies encryption. Missing support does not imply falsity.

## Review before gold use

The user must inspect every claim/page pair, verify extraction and source/page,
approve or correct NLI direction, relationship, relevance and rationale, and
confirm scope/time/precedence assumptions and pool-level conflicts. Record the
actual reviewer, review date and decisions in the companion review
record; do not merely rename provisional labels as human-reviewed. Preserve
synthetic provenance after review. No records in this seed are human verified.

Step 2 showed top-1 can hide conflicts and relative filtering can retain an
all-weak pool. These cases preserve those diagnostic shapes without carrying
over its score grid or selecting a threshold. This tiny, dependent, explicit
synthetic sample is for fixture development only. Before calibration or accuracy
claims, collect independently reviewed representative policies and held-out
examples, including harder contradictions and relevant neutral passages. Split
by policy/claim family to avoid leakage; do not tune and evaluate on these same
pairs. No dataset expansion or threshold experiment is part of this substep.

## Reproduce validation

From the repository root using the existing environment:

```powershell
& .\.venv\Scripts\python.exe -B -m pytest tests/test_evidence_selection_seed.py -q -p no:cacheprovider
& .\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider
git diff --check
```

Tests verify strict fixture fields, types, IDs, labels, review flags, synthetic
marking, relevance/conflict consistency, source snapshots, compact coverage and
rejection of malformed fixtures. They do not establish semantic correctness.
No models, downloads, Gemini calls or added dependencies are needed.

## Individual approval queue

`evidence_selection_review.json` contains one linked review record for each of
all 14 pairs. Every record is `pending_user_approval`: `reviewer`, `reviewed_at`
and `decision` are null, and `human_verified` is false. These are proposed
annotations awaiting decisions, not completed reviewer decisions. The `proposed`
block preserves claim, passage, source/page, relationship, relevance, rationale,
provenance (including synthetic marking) and pool conflict context. Tests compare
these fields against the seed so changed annotations cannot silently drift.

To review individually, open the seed and companion file side by side, then say,
for example: "Approve q002-baseline-p1 as proposed; reviewer: [your name],
reviewed at: [date]" or "Correct [evidence ID]: relationship ..., relevance ...,
rationale ...". Include source/page and conflict-assumption corrections if needed.
Approval covers the pair and its displayed context, including the same-scope,
same-time and no-precedence assumptions for constructed conflicts described above.
A later step can record those actual decisions with reviewer/date and extend the
pending-only schema and tests to support approved or corrected records. Do not
change flags alone or infer approval from this prepared queue. Synthetic passages
remain synthetic after approval; pair approval alone does not make this seed gold.
