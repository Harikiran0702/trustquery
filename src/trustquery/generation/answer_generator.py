from pathlib import Path
from trustquery.gemini_client import generate_text

PROMPT_PATH = Path("prompts/grounded_answer.txt")


def format_evidence(retrieved_chunks):
    evidence_blocks = []

    for index, chunk in enumerate(retrieved_chunks, start=1):
        metadata = chunk.get("metadata", {})

        source = metadata.get("source", "unknown")
        page = metadata.get("page", "unknown")
        text = chunk.get("text", "")

        evidence_blocks.append(
            f"[Evidence {index}]\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Text: {text}"
        )

    return "\n\n".join(evidence_blocks)


def build_grounded_prompt(question, retrieved_chunks):
    if not question.strip():
        raise ValueError("question cannot be empty")

    if not retrieved_chunks:
        raise ValueError("retrieved_chunks cannot be empty")

    template = PROMPT_PATH.read_text(encoding="utf-8")

    evidence = format_evidence(retrieved_chunks)

    return template.format(
        question=question,
        evidence=evidence,
    )
def generate_grounded_answer(question, retrieved_chunks):
    prompt = build_grounded_prompt(
        question=question,
        retrieved_chunks=retrieved_chunks,
    )

    return generate_text(prompt)