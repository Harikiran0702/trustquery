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

    answer = generate_text(prompt)

    if "Insufficient evidence" in answer:
        return "Insufficient evidence in the provided documents."

    has_source = "Source:" in answer
    has_page = "Page:" in answer

    if not (has_source and has_page):
        top_metadata = retrieved_chunks[0].get("metadata", {})
        source = top_metadata.get("source", "unknown")
        page = top_metadata.get("page", "unknown")

        answer = (
            f"{answer.strip()}\n\n"
            f"Source: {source}, Page: {page}"
        )

    return answer