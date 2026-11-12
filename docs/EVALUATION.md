# Evaluation

Run `python scripts/evaluate.py --hybrid` for the 100 provisional calibration records. Add `--include-holdout` only after freezing the pipeline and tests. Both columns use offline policy extracts. Hybrid uses the two pinned local models. To populate the cache explicitly, run `python scripts/download_models.py`.

| Classification (count / drafted / abstained) | BM25 | Hybrid |
| --- | ---: | ---: |
| policy-supported | 54 / 54 / 0 | 54 / 51 / 3 |
| partial | 18 / 16 / 2 | 18 / 16 / 2 |
| policy-only-not-implemented | 15 / 14 / 1 | 15 / 14 / 1 |
| unsupported/TBD | 12 / 4 / 8 | 12 / 5 / 7 |
| explicit-negative | 1 / 0 / 1 | 1 / 0 / 1 |

Page metrics use the 54 draft-eligible records with source and page labels. Candidate ranks are measured before answer filtering. A target outside the returned candidates contributes zero.

| Retrieval metric | BM25 | Hybrid |
| --- | ---: | ---: |
| Page hit@1 | 0.481481 | 0.611111 |
| Page hit@3 | 0.685185 | 0.740741 |
| Page MRR | 0.617383 | 0.703042 |
| Document hit@5 | 0.907407 | 0.962963 |

## Unseen questions

Aggregate results from one pass after freezing the implementation and tests. Expected decisions come from `metadata.should_answer`.

| Count | BM25 | Hybrid |
| --- | ---: | ---: |
| Labeled questions | 20 | 20 |
| Unrecognized records | 0 | 0 |
| Expected to draft | 12 | 12 |
| Expected to abstain | 8 | 8 |
| Drafted | 12 | 11 |
| Abstained | 8 | 9 |
| Drafted when expected to draft | 11 | 11 |
| Abstained when expected to abstain | 7 | 8 |
