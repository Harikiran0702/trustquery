# TrustQuery

[![CI](https://github.com/Harikiran0702/trustquery/actions/workflows/ci.yml/badge.svg)](https://github.com/Harikiran0702/trustquery/actions/workflows/ci.yml)

TrustQuery is an experimental security-questionnaire evidence and human-review application. Its CloudDesk documents are fictional, model-authored proposed policies. They establish no implemented controls, compliance, or operational proof.

The established retrieval pipeline combines BM25, MiniLM dense embeddings, reciprocal rank fusion (RRF), and optional cross-encoder reranking. Offline batch processing defaults to BM25 and conservative abstention. Experimental policy generation and section matching require explicit opt-in; candidate references are unverified. Human approval and final citations must not be inferred from model output. NLI scores are experimental and uncalibrated.

Generation code uses **Gemini**, with `GEMINI_API_KEY` in a local ignored `.env`. Generation sends supplied evidence to that service when explicitly invoked. The configured Gemini model and API availability have not been verified during release hardening. The offline workflow and test runner make no external model/API calls.

## Install and run

Use Python 3.12 and a source checkout, with PowerShell at the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[review,test,authoring]"
.\.venv\Scripts\python.exe -m pytest -m "not model" -q
.\.venv\Scripts\python.exe -m trustquery.batch data/eval/clouddesk_questionnaire.json "$env:TEMP/trustquery-drafts-new.json"
.\.venv\Scripts\python.exe -m streamlit run src/trustquery/review_app.py --server.address 127.0.0.1 --server.maxUploadSize 1 --browser.gatherUsageStats false
```

Install may require package downloads; a fully offline install requires a separately provisioned wheelhouse (`--no-index --find-links PATH`). ReportLab is an optional authoring extra; it is not required to read existing PDFs. Review UI starts empty: upload a UTF-8 questionnaire CSV/JSON, or reopen a paired saved results JSON and SQLite database in the sidebar. Arbitrary policy PDF upload is unsupported. The upload workflow uses only eight hash-checked bundled synthetic proposed policies, offline BM25, conservative answers and explicit human decisions; Gemini generation is disabled in this workflow. See [v1 upload workflow](docs/questionnaire_upload.md), [batch processing](docs/batch_questionnaires.md), [human review](docs/human_review.md), and [Excel export](docs/excel_export.md). Choose fresh output names; do not overwrite historical batches.

Model loading is lazy and local-only by default. Dense retrieval requires cached `sentence-transformers/all-MiniLM-L6-v2`; reranking and NLI require their separate caches. Exact observed revisions are in `src/trustquery/model_settings.py`. Missing cache files fail rather than download. BM25 needs no model cache. Model provisioning is a separate explicit network operation; no cache was downloaded for this step.

### Draft pipeline and offline cache

With no `GEMINI_API_KEY` environment variable, the pipeline reads
`data/demo/answers_cache.json`, checking its extracts against current evidence.
Missing or stale entries fall back to original policy extracts. Rebuild all 100
sample entries with `python scripts/build_demo_cache.py` in the installed project
environment. This command uses policy PDFs and questionnaire text only, makes no
API calls or model downloads, and records empty answers with reasons when evidence
is absent. Cache entries are offline extracts, not Gemini outputs. The application
reads keys from the process environment and does not automatically load `.env`.
