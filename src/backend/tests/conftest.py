import pytest
import numpy as np


@pytest.fixture
def sample_audio_data():
    return {
        "audio": np.random.randn(22050),
        "sample_rate": 22050,
        "duration": 1.0,
        "filename": "test.mp3",
        "genre": "techno",
        "path": "/fake/path/test.mp3",
    }


@pytest.fixture
def sample_features():
    return np.random.randn(45)


@pytest.fixture
def sample_metadata():
    return {
        "filename": "test.mp3",
        "genre": "techno",
        "duration": 1.0,
        "path": "/fake/path/test.mp3",
    }
