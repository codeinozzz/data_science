import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.ingestion.audio_processor import AudioProcessor


@pytest.fixture
def sample_audio():
    return {
        "audio": np.random.randn(22050),
        "sample_rate": 22050,
        "duration": 1.0,
        "filename": "test.mp3",
        "genre": "techno",
        "path": "/fake/path/test.mp3",
    }


@pytest.fixture
def processor():
    return AudioProcessor()


def test_extract_features_shape(processor, sample_audio):
    result = processor.extract_features(sample_audio)
    assert result["features"].shape == (45,)


def test_extract_features_metadata(processor, sample_audio):
    result = processor.extract_features(sample_audio)
    metadata = result["metadata"]

    assert metadata["filename"] == "test.mp3"
    assert metadata["genre"] == "techno"
    assert metadata["duration"] == 1.0


def test_extract_features_no_nan(processor, sample_audio):
    result = processor.extract_features(sample_audio)
    assert not np.isnan(result["features"]).any()


def test_process_batch(processor):
    audio_list = [
        {
            "audio": np.random.randn(22050),
            "sample_rate": 22050,
            "duration": 1.0,
            "filename": f"test{i}.mp3",
            "genre": "techno",
            "path": f"/fake/path/test{i}.mp3",
        }
        for i in range(3)
    ]

    results = processor.process_batch(audio_list)
    assert len(results) == 3
    assert all(r["features"].shape == (45,) for r in results)
