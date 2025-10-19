import pytest
from pathlib import Path
from fastapi import HTTPException
import sys
import tempfile

sys.path.append(str(Path(__file__).parent.parent))
from utils.security import sanitize_filename, safe_path, validate_file_content


def test_sanitize_filename():
    assert sanitize_filename("test.mp3") == "test.mp3"
    assert sanitize_filename("test file.mp3") == "test file.mp3"
    assert ".." not in sanitize_filename("../etc/passwd")
    assert sanitize_filename("test<>?.mp3") == "test.mp3"


def test_safe_path_valid():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        result = safe_path(base, "test.mp3")
        assert str(result).startswith(str(base))


def test_safe_path_traversal_attack():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        with pytest.raises(HTTPException):
            safe_path(base, "../../../etc/passwd")


def test_validate_file_content_empty():
    with pytest.raises(HTTPException) as exc:
        validate_file_content(b'', ['.mp3'])
    assert exc.value.status_code == 400


def test_validate_file_content_mp3():
    mp3_header = b'ID3' + b'\x00' * 100
    assert validate_file_content(mp3_header, ['.mp3']) == True


def test_validate_file_content_wav():
    wav_header = b'RIFF' + b'\x00' * 4 + b'WAVE' + b'\x00' * 100
    assert validate_file_content(wav_header, ['.wav']) == True