from pathlib import Path


def load_text_file(file_path: str) -> dict:
    """Load a UTF-8 text file and return its content with source metadata."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    text = path.read_text(encoding="utf-8")

    return {
        "text": text,
        "source": path.name,
    }