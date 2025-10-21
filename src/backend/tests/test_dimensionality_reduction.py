import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.models.dimensionality_reduction import (
    DimensionalityReducer,
    reduce_to_2d,
    reduce_to_3d,
)


@pytest.fixture
def sample_embeddings():
    """Create sample 45D embeddings"""
    np.random.seed(42)
    return np.random.randn(100, 45)


def test_tsne_2d_reduction(sample_embeddings):
    reducer = DimensionalityReducer(method="tsne", n_components=2)
    result = reducer.fit_transform(sample_embeddings)

    assert result.shape == (100, 2)
    assert reducer.embedding is not None


def test_tsne_3d_reduction(sample_embeddings):
    reducer = DimensionalityReducer(method="tsne", n_components=3)
    result = reducer.fit_transform(sample_embeddings)

    assert result.shape == (100, 3)


def test_reduce_to_2d_helper(sample_embeddings):
    result = reduce_to_2d(sample_embeddings, method="tsne")

    assert result.shape == (100, 2)
    assert not np.isnan(result).any()


def test_reduce_to_3d_helper(sample_embeddings):
    result = reduce_to_3d(sample_embeddings, method="tsne")

    assert result.shape == (100, 3)
    assert not np.isnan(result).any()


def test_save_load_embeddings(sample_embeddings, tmp_path):
    reducer = DimensionalityReducer(method="tsne", n_components=2)
    reducer.fit_transform(sample_embeddings)

    filepath = tmp_path / "embeddings_2d.npy"
    reducer.save(filepath)

    loaded = DimensionalityReducer.load(filepath)

    assert loaded.embedding.shape == (100, 2)
    np.testing.assert_array_equal(reducer.embedding, loaded.embedding)
