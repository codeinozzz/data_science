import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.embeddings.audio_embedder import AudioEmbedder


@pytest.fixture
def embedder():
    return AudioEmbedder()


@pytest.fixture
def processed_audio():
    return {"features": np.random.randn(45), "metadata": {"filename": "test.mp3"}}


def test_generate_embedding_shape(embedder, processed_audio):
    embedding = embedder.generate_embedding(processed_audio)
    assert embedding.shape == (45,)


def test_generate_embedding_normalized(embedder, processed_audio):
    embedding = embedder.generate_embedding(processed_audio)
    mean = np.mean(embedding)
    std = np.std(embedding)

    assert abs(mean) < 0.1
    assert abs(std - 1.0) < 0.1


def test_generate_embeddings_batch(embedder):
    batch = [
        {"features": np.random.randn(45), "metadata": {"filename": f"test{i}.mp3"}}
        for i in range(5)
    ]

    embeddings = embedder.generate_embeddings_batch(batch)
    assert len(embeddings) == 5
    assert all(e.shape == (45,) for e in embeddings)


def test_get_feature_dimension(embedder, processed_audio):
    embedder.generate_embedding(processed_audio)
    assert embedder.get_feature_dimension() == 45
