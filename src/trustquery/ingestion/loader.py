from pathlib import Path
import re

from pypdf import PdfReader


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def load_text_file(file_path: str) -> dict:
    """Load a UTF-8 text file and return its content with source metadata."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    text = path.read_text(encoding="utf-8")

    return {
        "text": _normalize_whitespace(text),
        "source": path.name,
    }


def load_pdf(file_path: str) -> list[dict]:
    """Load a PDF and return extracted text with source and page metadata."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    reader = PdfReader(path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text and text.strip():
            pages.append(
                {
                    "text": _normalize_whitespace(text),
                    "source": path.name,
                    "page": page_number,
                }
            )

    return pages