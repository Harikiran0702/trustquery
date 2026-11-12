# Excel review export (Major Deliverable 3)

In the UI, save explicit decisions, then choose **Prepare reviewed Excel** and
**Download reviewed Excel**. Unsaved work blocks preparation/download; export
contains all saved questions regardless of current filters. A prepared download
is a snapshot: prepare again after concurrent changes. Browser download delivery
remains unverified for Step 6; AppTest preparation and openpyxl reading are tested.
See [questionnaire upload](questionnaire_upload.md).

For CLI export, from PowerShell at the repository root, substitute the saved run
paths and a fresh output filename in this historical command example:

```powershell
.\.venv\Scripts\python.exe -m trustquery.review_export data/batch_runs/major_deliverable_1/results.json data/reviews/major_deliverable_2.sqlite3 data/reviews/review_export.xlsx
```

Use a fresh output filename in an existing directory. Existing output is never
replaced. The SQLite database must already exist and match the exact SHA-256 of
the batch bytes. Export opens it read-only and reads hash binding plus all latest
saved revisions in one transaction. No retrieval, generation, source PDF reads,
review saves or evaluation changes occur. CLI integration deliberately exports
saved state independently of browser filters, stale displayed revisions and
unsaved/conflicted drafts. Save or resolve drafts in the review UI first when
you want them included. A concurrent commit after the read snapshot starts is
included on the next export, not partially mixed into this one.

## Workbook fields

- **Questions** retains batch order and includes IDs, full questions, effective
  saved status, saved reviewed answer, a separate approved-policy-extract field,
  approval scope, missing implementation-evidence flag, reviewer, rationale,
  revision, SQLite UTC save timestamp, scope attestation, original outcome and
  batch draft. JSON columns retain candidate selections, complete source
  provenance, assessment, errors, original stored status, stored and batch
  citations (explicitly unverified), and the full latest saved review payload.
- **Verified citations** has headers only. The current review model has no
  citation-verification operation. Neither batch citations nor stored citation
  fields can populate this sheet.
- **UNVERIFIED candidates** contains every original candidate with selection
  indicator, exact source, one-based PDF page, zero-based chunk index, chunk ID,
  full quote and complete reference JSON. Selection and extract approval never
  change its UNVERIFIED CANDIDATE status.
- **Export notes** records the batch hash and interpretation limits.

Absent reviews have revision 0, status unreviewed and a blank reviewed answer;
the original batch answer stays in its separately labeled unreviewed column.
Edited, rejected and needs-evidence answers remain available for review but
never populate approved-policy-extract. Approved state is rechecked against the
existing exact policy-extract guard, scope, rationale and assessment; inconsistent
or unknown status is labeled invalid_stored_state and cannot produce an approved
extract. This is a consistency check, not semantic validation or tamper detection.
An approval covers only synthetic policy text, never operating controls or
compliance. Operational proof is always false.

Implementation-evidence-missing is true when implementation_required is true or
absent/unknown, when the batch outcome requires implementation evidence, or when
the saved status is needs_evidence. False does not demonstrate implementation or
completeness. Processing failures retain their original outcome and errors.

## Safety and reproducibility

Every cell is an explicit OOXML inline string, including formula-like strings,
URLs, JSON, timestamps and numeric locators. There are no formulas, external
links or macros; text is preserved without an apostrophe prefix. The workbook
has frozen headers, column widths and filters. Long text can be inspected in
Excel's formula bar; this is a wide detailed export rather than a printed report.
Cells exceeding Excel's 32,767 UTF-16-unit text limit, XML-incompatible text and
worksheet size overflow fail without truncation. The output is assembled before
opening the destination; validation failures do not leave a partial workbook.
No parent folders are created automatically.

Identical batch bytes and latest saved state produce byte-identical XLSX output
in the same Python environment: fixed ZIP timestamps, stable part ordering and
canonical JSON. There is no generated-at timestamp. Save timestamps come from
stored events. Local source PDFs are still not hash-bound to retrieval, and no
PDF or citation verification is introduced. Export is not a signed audit service
and does not include the full revision history (available in SQLite/UI).

## Validation and example

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[review,export-test]"
.\.venv\Scripts\python.exe -m pytest tests/test_review_export.py tests/test_review.py tests/test_review_ui_reliability.py tests/test_review_evidence.py -q
.\.venv\Scripts\python.exe -m pytest -q
```

`docs/examples/major_deliverable_3.xlsx` exports the existing 100-question batch
and the existing review database, which had no saved review events when generated.
It intentionally has 100 unreviewed rows, 500 unverified candidates and zero
verified citations. Tests use separate temporary databases for edited, approved,
rejected, incomplete, unknown and conflicting state, including WAL concurrency.
Independent openpyxl reopening checks workbook structure and literal cells.
Native Microsoft Excel rendering has not been asserted by those tests.
