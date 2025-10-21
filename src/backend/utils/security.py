from pathlib import Path
from fastapi import HTTPException
import re


def sanitize_filename(filename: str) -> str:
    """
    Remove potentially dangerous characters from filename.
    """
    safe_name = re.sub(r"[^\w\s.-]", "", filename)
    safe_name = safe_name.replace("..", "")
    return safe_name


def safe_path(base_dir: Path, filename: str) -> Path:
    """
    Ensure path is within base_dir to prevent path traversal attacks.
    """
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(400, "Invalid file path")

    clean_name = sanitize_filename(filename)
    target = (base_dir / clean_name).resolve()

    try:
        target.relative_to(base_dir.resolve())
    except ValueError:
        raise HTTPException(400, "Invalid file path")

    return target


def validate_file_content(content: bytes, expected_extensions: list) -> bool:
    """
    Basic validation of file content.
    """
    if len(content) == 0:
        raise HTTPException(400, "Empty file")

    magic_numbers = {b"ID3": [".mp3"], b"RIFF": [".wav"], b"fLaC": [".flac"]}

    for magic, exts in magic_numbers.items():
        if content.startswith(magic):
            return True

    if content[:4] == b"RIFF" and content[8:12] == b"WAVE":
        return True

    return False
