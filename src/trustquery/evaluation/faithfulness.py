import re


def extract_claims(answer: str) -> list[str]:
    """Split basic sentences, excluding standalone Source/Page citation lines."""
    content = "\n".join(
        line for line in answer.splitlines()
        if not re.match(r"^\s*(?:Source|Page)\s*:", line, re.IGNORECASE)
    )
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", content.strip())
        if sentence.strip()
    ]


def faithfulness_score(
    claims: list[str], supported_claims: list[bool]
) -> float:
    """Return the proportion of claims marked as supported by evidence."""
    if len(claims) != len(supported_claims):
        raise ValueError("claims and supported_claims must have equal lengths")
    if not claims:
        return 0.0
    return sum(supported_claims) / len(claims)
