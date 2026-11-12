import copy
import hashlib
import json
from pathlib import Path
import pytest
from pypdf import PdfReader
from trustquery.experimental_policy_sections import generate_section_draft
ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture(scope="module")
def inputs():
    config=json.loads((ROOT/"src/trustquery/experimental_policy_plans.json").read_text())["corpus_sha256"]
    corpus={n:(ROOT/"data/sample"/n).read_bytes() for n in config}
    source="change_management_policy.pdf"
    text=PdfReader(ROOT/"data/sample"/source).pages[2].extract_text()
    return "What requirements does the policy state for Engineering safeguards and testing?", [dict(source=source,page=3,text=text)], corpus

def test_complete_context_and_binding(inputs):
    q,p,c=inputs
    original=copy.deepcopy(inputs)
    out=generate_section_draft(q,p,c,enabled=True)
    assert out["answer"],out["reason"]
    assert inputs==original
    assert len(out["claims"])==6
    assert sum(x["role"]=="requested_section" for x in out["claims"])==1
    assert out["citations"]==[]
    assert not out["human_approval"] and not out["operational_proof"]
    for claim,ref in zip(out["claims"],out["validated_policy_citations"]):
        assert claim["text"]==ref["quote"]
        assert ref["pdf_sha256"]==hashlib.sha256(c[ref["source"]]).hexdigest()

@pytest.mark.parametrize("q",[
    "Do you implement Engineering safeguards and testing?",
    "What requirements does the policy state for Engineering safeguards and testing? Is it deployed?",
    "What requirements does the policy state for Engineering safeguards and testing and approvals?",
    "What requirements does the policy state for TLS 1.3 only?",
    "What requirements does the policy state for no encryption?",
    "What requirements does the policy state for rotation?",
    "What requirements does the policy state for unknown?",
])
def test_ambiguous_missing_operational_refuse(inputs,q):
    _,p,c=inputs
    assert not generate_section_draft(q,p,c,enabled=True)["answer"]

@pytest.mark.parametrize("mutation",["disabled","empty","truncate","forge","page","version","unrelated","missing","provider","drop_context","forge_claim_ref"])
def test_fail_closed(inputs,mutation):
    q,p,c=copy.deepcopy(inputs)
    kw=dict(enabled=True)
    if mutation=="disabled":
        kw["enabled"]=False
    if mutation=="empty":
        p=[]
    if mutation=="truncate":
        p[0]["text"]="Engineering safeguards and testing"
    if mutation=="forge":
        p[0]["text"]+=" invented requirement"
    if mutation=="page":
        p[0]["page"]=2
    if mutation=="version":
        c[p[0]["source"]]+=b" changed"
    if mutation=="unrelated":
        c["access_control_policy.pdf"]+=b" changed"
    if mutation=="missing":
        c.pop("access_control_policy.pdf")
    if mutation in {"provider","drop_context","forge_claim_ref"}:
        def provider(prompt):
            obj=json.loads(prompt)
            if mutation=="provider":
                obj["claims"][0]["text"]="Approved and implemented"
            if mutation=="drop_context":
                obj["claims"].pop()
            if mutation=="forge_claim_ref":
                obj["claims"][0]["evidence_id"]="other:p1"
            return obj
        kw["generator"]=provider
    out=generate_section_draft(q,p,c,**kw)
    assert not out["answer"] and not out["claims"] and not out["validated_policy_citations"]

@pytest.mark.parametrize("source,page,title,needles",[
    ("encryption_key_management_policy.pdf",2,"Encryption in transit and at rest",["TLS 1.2 or TLS 1.3","do not accept","90 days","approval not"]),
    ("encryption_key_management_policy.pdf",3,"Key storage, access and lifecycle",["non-exportable","when","service requires it","no default"]),
    ("backup_disaster_recovery_policy.pdf",5,"Restore testing, continuity exercises and dependencies",["quarterly","representative","annual"]),
])
def test_qualifications_are_lossless(inputs,source,page,title,needles):
    _,_,corpus=inputs
    text=PdfReader(ROOT/"data/sample"/source).pages[page-1].extract_text()
    out=generate_section_draft(f"What requirements does the policy state for {title}?",
        [dict(source=source,page=page,text=text)],corpus,enabled=True)
    assert out["answer"],out["reason"]
    normalized=" ".join(out["answer"].split()).lower()
    for needle in needles:
        assert needle.lower() in normalized
    assert out["support_status"]=="not_semantically_adjudicated"

