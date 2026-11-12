"""SHA-256 helpers for file provenance."""
import hashlib
from pathlib import Path


def text_sha256(data: bytes) -> str:
    """Hash text bytes with CRLF line endings normalized to LF."""
    return hashlib.sha256(data.replace(b'\r\n', b'\n')).hexdigest()


def legacy_crlf_sha256(data: bytes) -> str:
    """Reproduce historical text pins written with Windows line endings."""
    return hashlib.sha256(data.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')).hexdigest()


def file_sha256(path: str | Path, data: bytes) -> str:
    if Path(path).suffix.lower() in {'.json', '.txt', '.md', '.csv', '.toml'}:
        return text_sha256(data)
    return hashlib.sha256(data).hexdigest()
