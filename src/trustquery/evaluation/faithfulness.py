import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from trustquery.evaluation.nli import NLIBackend, NLIResult, verify_claim


def extract_claims(answer: str) -> list[str]:
    """Extract basic statements, skipping citation lines, questions and bare replies.

    This is a sentence-based heuristic, not a factuality or entailment check.
    """
    citation_line = re.compile(
        r"^\s*(?:\*\*|__|\*|_)?(?:Source|Page)"
        r"(?:\*\*|__|\*|_)?\s*:(?:\*\*|__|\*|_)?",
        re.IGNORECASE,
    )
    content = "\n".join(
        line for line in answer.splitlines() if not citation_line.match(line)
    )
    # Match explicit source/page pairs only, preserving surrounding facts.
    emphasis = r"(?:\*\*|__|\*|_)?"
    source = rf"{emphasis}Source{emphasis}\s*:{emphasis}\s*"
    page = rf"{emphasis}Page{emphasis}\s*:{emphasis}\s*\d+{emphasis}"
    pair = rf"{source}[^\n()\[\]]+?,\s*{page}"
    content = re.sub(
        rf"\(\s*{pair}\s*\)|\[\s*{pair}\s*\]", "", content,
        flags=re.IGNORECASE,
    )
    # Unwrapped citations require a filename to avoid consuming ordinary prose.
    parts = re.split(r"(\([^\n()]*\)|\[[^\n\[\]]*\])", content)
    content = "".join(
        part if part.startswith(("(", "[")) else re.sub(
            rf"{source}[\w./\\ -]+\.pdf\s*,\s*{page}(?![\w-])",
            "", part, flags=re.IGNORECASE,
        )
        for part in parts
    )
    content = re.sub(r"[ \t]+([.!?,;:])", r"\1", content)
    content = re.sub(r"[ \t]{2,}", " ", content)
    claims = []
    for sentence in re.split(r"(?<=[.!?])\s+", content.strip()):
        sentence = sentence.strip()
        # Ignore Markdown emphasis when identifying questions and bare replies.
        plain = sentence.strip("*_ ")
        if not plain or plain.endswith("?"):
            continue
        if re.fullmatch(r"yes[.!]*", plain, re.IGNORECASE):
            continue
        if re.fullmatch(
            r"(?:no|n/a|na|unknown)", sentence.strip("*_ .!?,:;"), re.IGNORECASE
        ):
            continue
        claims.append(sentence)
    return claims


def faithfulness_score(
    claims: list[str], supported_claims: list[bool]
) -> float:
    """Return the proportion of claims marked as supported by evidence."""
    if len(claims) != len(supported_claims):
        raise ValueError("claims and supported_claims must have equal lengths")
    if not claims:
        return 0.0
    return sum(supported_claims) / len(claims)


@dataclass(frozen=True)
class ChunkVerification:
    evidence_index: int
    metadata: dict[str, Any]
    result: NLIResult


@dataclass(frozen=True)
class ClaimVerification:
    claim: str
    supported: bool
    evidence: tuple[ChunkVerification, ...]
    conflicted: bool = False


@dataclass(frozen=True)
class AnswerFaithfulness:
    score: float
    claims: tuple[ClaimVerification, ...]


def answer_faithfulness(
    answer: str,
    evidence_chunks: Sequence[Mapping[str, Any]],
    backend: NLIBackend,
) -> AnswerFaithfulness:
    """Verify extracted claims against each retrieved chunk independently.

    The caller supplies the relevant retrieved set; no further relevance filter
    is applied. A claim is conflicted when chunks both entail and contradict it.
    Only entailment without contradiction supports a claim; all verdicts and
    source/page/chunk metadata remain available for inspection.
    No evidence means unsupported claims. No claims means score zero, not success.

    Validate all chunks before inference, even for an empty answer. Metadata may
    be absent but must be a mapping when present. Blank text is invalid evidence.
    Backend failures (including token overflow) abort with pair context and the
    original exception as cause: no truncation, skipping, or partial score.
    Custom backends are responsible for rejecting inputs beyond their token limit.
    Claim extraction is heuristic; NLI probabilities are not calibrated confidence.
    """
    if not isinstance(answer, str):
        raise TypeError("answer must be a string")
    if not isinstance(evidence_chunks, Sequence) or isinstance(
        evidence_chunks, (str, bytes)
    ):
        raise TypeError("evidence_chunks must be a sequence of chunk mappings")
    chunks = []
    for index, chunk in enumerate(evidence_chunks):
        if not isinstance(chunk, Mapping):
            raise ValueError(f"evidence chunk {index} must be a mapping")
        text = chunk.get("text")
        metadata = chunk.get("metadata", {})
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"evidence chunk {index} needs non-empty string text")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"evidence chunk {index} metadata must be a mapping")
        chunks.append((text, dict(metadata)))

    claims = extract_claims(answer)
    results = []
    for claim_index, claim in enumerate(claims):
        evidence = []
        for index, (text, metadata) in enumerate(chunks):
            try:
                verdict = verify_claim(claim, text, backend)
            except Exception as exc:
                raise RuntimeError(
                    f"NLI verification failed for claim {claim_index}, "
                    f"evidence chunk {index}: {exc}"
                ) from exc
            evidence.append(ChunkVerification(index, dict(metadata), verdict))
        entailed = any(item.result.label == "entailment" for item in evidence)
        contradicted = any(item.result.label == "contradiction" for item in evidence)
        results.append(ClaimVerification(
            claim, entailed and not contradicted, tuple(evidence),
            conflicted=entailed and contradicted,
        ))
    return AnswerFaithfulness(
        faithfulness_score(claims, [item.supported for item in results]),
        tuple(results),
    )
