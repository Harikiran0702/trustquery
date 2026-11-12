"""Opt-in section extraction experiment. No semantic entailment or runtime integration."""
import copy
import hashlib
import io
import json
import re
from pypdf import PdfReader
from trustquery.experimental_generation import BOUNDARY, digest
from trustquery.experimental_policy_generation import PLAN_PATH, norm

INTENT = re.compile(r"What requirements does the policy state for ([A-Za-z][A-Za-z0-9 ,/&()-]{2,120})\?")

def generate_section_draft(question, passages, corpus, *, enabled=False, generator=None):
    """Interpret only an explicit complete-section intent; never guess missing scope.

    This is structural requirement extraction, not arbitrary sentence selection.
    Every source page accompanies the target section as mandatory context.
    Existing inspected plans and original questionnaire behavior are untouched.
    """
    out = dict(experimental=True, method="bound_complete_section_v1",
        status="needs_evidence", answer="", claims=[], citations=[],
        validated_policy_citations=[], operational_proof=False, human_approval=False,
        source_approval="not_recorded", review_status="requires_human_review")
    if enabled is not True:
        out["reason"] = "Explicit experimental opt-in required"
        return out
    try:
        match = INTENT.fullmatch(question)
        if not match:
            raise ValueError("Unsupported, ambiguous or operational intent")
        title = match[1].casefold()
        config = json.loads(PLAN_PATH.read_text(encoding="utf-8"))["corpus_sha256"]
        if set(corpus) != set(config):
            raise ValueError("Complete pinned corpus required")
        pages = {}
        for name, expected in config.items():
            data = corpus[name]
            if not isinstance(data, bytes) or hashlib.sha256(data).hexdigest() != expected:
                raise ValueError("Source version changed; independent inspection required")
            pages[name] = [p.extract_text() or "" for p in PdfReader(io.BytesIO(data)).pages]
        # Numbered policy sections must start on a fresh page. Fail closed on
        # unrecognized/continued layouts instead of treating a sentence as a section.
        sections = []
        for name, texts in pages.items():
            for index, text in enumerate(texts, 1):
                headings = list(re.finditer(r"(?m)^([1-9][0-9]*)\. ([^\n]+)$", text))
                if len(headings) == 1:
                    h = headings[0]
                    if norm(h[2]).casefold() == title:
                        sections.append((name, index, text[h.start():]))
        if len(sections) != 1:
            raise ValueError("Missing or ambiguous complete section title")
        source, target, section = sections[0]
        for expected_number, text in enumerate(pages[source], 1):
            headings = re.findall(r"(?m)^([1-9][0-9]*)\. ([^\n]+)$", text)
            if len(headings) != 1 or int(headings[0][0]) != expected_number:
                raise ValueError("Unsupported section layout; complete context cannot be established")
        if not isinstance(passages, list) or not passages:
            raise ValueError("Selected evidence required")
        anchored = False
        for passage in passages:
            name, page, text = passage["source"], passage["page"], passage["text"]
            if (name not in pages or type(page) is not int or
                not 1 <= page <= len(pages[name]) or not isinstance(text, str) or
                not norm(text) or norm(text) not in norm(pages[name][page-1])):
                raise ValueError("Unbound evidence locator/text")
            if name == source and page == target and norm(section) in norm(text):
                anchored = True
        if not anchored:
            raise ValueError("Complete selected section required")
        # Every page is retained, in document order. The target is a quoted
        # requirement block, others are mandatory scope/status/exception context.
        refs, claims = [], []
        for page, text in enumerate(pages[source], 1):
            ref_id = f"{source}:p{page}"
            refs.append(dict(evidence_id=ref_id, source=source, page=page,
                quote=text, pdf_sha256=config[source], page_sha256=digest(text),
                approval_status="not_recorded", validation="local_pdf_bytes_and_page",
                independently_verified=False))
            claims.append(dict(evidence_id=ref_id,
                role="requested_section" if page == target else "mandatory_document_context",
                text=text))
        proposal = generator(json.dumps(dict(claims=claims))) if generator else dict(claims=copy.deepcopy(claims))
        if isinstance(proposal, str):
            proposal = json.loads(proposal)
        if not isinstance(proposal, dict) or set(proposal) != {"claims"} or proposal["claims"] != claims:
            raise ValueError("Changed claim, context or binding")
        out.update(status="experimental_draft", claims=claims,
            validated_policy_citations=refs, corpus_sha256=config,
            question_intent=dict(kind="complete_policy_section", title=match[1]),
            support_status="not_semantically_adjudicated",
            coverage="Complete quoted section with full document context; no semantic completeness inference",
            answer="POLICY TEXT EXTRACTION (synthetic, proposed; approval not recorded)\n"+
                "\n".join(f"[{c['role']}; {c['evidence_id']}]\n{c['text']}" for c in claims)+"\n"+BOUNDARY,
            reason="Complete section and mandatory context bound to pinned PDF bytes")
    except Exception as exc:
        out["reason"] = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
    return out

