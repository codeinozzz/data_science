import pytest
from fastapi import HTTPException
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.validators import validate_audio_file, validate_search_params


def test_validate_audio_file_valid():
    assert validate_audio_file("test.mp3", 1024 * 1024) == True
    assert validate_audio_file("test.wav", 5 * 1024 * 1024) == True
    assert validate_audio_file("test.flac", 100 * 1024) == True


def test_validate_audio_file_invalid_extension():
    with pytest.raises(HTTPException) as exc:
        validate_audio_file("test.txt", 1024)
    assert exc.value.status_code == 400


def test_validate_audio_file_too_large():
    with pytest.raises(HTTPException) as exc:
        validate_audio_file("test.mp3", 20 * 1024 * 1024)
    assert exc.value.status_code == 413


def test_validate_search_params_valid():
    assert validate_search_params(5) == True
    assert validate_search_params(1) == True
    assert validate_search_params(20) == True


def test_validate_search_params_invalid():
    with pytest.raises(HTTPException):
        validate_search_params(0)

    with pytest.raises(HTTPException):
        validate_search_params(21)
