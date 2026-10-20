import re


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
