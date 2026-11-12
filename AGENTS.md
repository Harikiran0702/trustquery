# AGENTS.md

## Project goal
TrustQuery is a PORTFOLIO PROJECT: a RAG copilot for security questionnaires.
Pipeline: hybrid retrieval (BM25 + MiniLM + RRF + reranker) -> grounded answer
with citations -> faithfulness check -> confidence label -> human review -> Excel export.
Optimize for: working features, clear code, measurable results, a clean demo.

## Rules
- Make the smallest change that completes the task. No unrelated refactors.
- Do NOT create new report, audit, validation, preservation, or "stepN" files.
  Only allowed docs: README.md, docs/DESIGN.md, docs/EVALUATION.md.
- Do NOT add disclaimers beyond one "Demo uses fictional policies" notice.
- Do NOT add allowlists, hash pins, lockfile gates, or release-hardening machinery.
- Commits on private branches need no audit or allowlist; just keep .env and secrets out.
- Prefer a draft answer with a confidence label over refusing. Human approval is the safety layer.
- Tests needing torch or downloaded models: mark @pytest.mark.model, skip when unavailable.
- Codex must not open, print, or commit .env. The app itself may load it with python-dotenv.
- Always run the FULL suite `pytest -m "not model" -q`. Never report a selected subset as passing.
- Final message: short summary of what changed (max 10 lines).
