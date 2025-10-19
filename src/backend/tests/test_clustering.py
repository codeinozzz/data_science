import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.models.clustering import AudioClusterer, find_optimal_k, analyze_clusters


@pytest.fixture
def sample_embeddings():
    """Create synthetic embeddings for testing"""
    np.random.seed(42)
    cluster1 = np.random.randn(30, 45) + [1, 0, 0, 0, 0] * 9
    cluster2 = np.random.randn(30, 45) + [-1, 0, 0, 0, 0] * 9
    cluster3 = np.random.randn(30, 45) + [0, 1, 0, 0, 0] * 9
    return np.vstack([cluster1, cluster2, cluster3])


@pytest.fixture
def sample_metadata():
    """Create sample metadata"""
    metadata = []
    for i in range(90):
        genre = 'techno' if i < 30 else ('house' if i < 60 else 'ambient')
        metadata.append({
            'filename': f'sample_{i}.mp3',
            'genre': genre
        })
    return metadata


def test_kmeans_clustering(sample_embeddings):
    clusterer = AudioClusterer(method='kmeans', n_clusters=3)
    clusterer.fit(sample_embeddings)
    
    assert clusterer.labels is not None
    assert len(clusterer.labels) == len(sample_embeddings)
    assert len(np.unique(clusterer.labels)) == 3


def test_dbscan_clustering(sample_embeddings):
    clusterer = AudioClusterer(method='dbscan')
    clusterer.fit(sample_embeddings)
    
    assert clusterer.labels is not None
    assert len(clusterer.labels) == len(sample_embeddings)


def test_clustering_metrics(sample_embeddings):
    clusterer = AudioClusterer(method='kmeans', n_clusters=3)
    clusterer.fit(sample_embeddings)
    
    assert 'silhouette' in clusterer.metrics
    assert 'n_clusters' in clusterer.metrics
    assert clusterer.metrics['silhouette'] > 0


def test_find_optimal_k(sample_embeddings):
    results = find_optimal_k(sample_embeddings, k_range=(2, 5))
    
    assert 'k_values' in results
    assert 'silhouette_scores' in results
    assert len(results['k_values']) == 4


def test_analyze_clusters(sample_embeddings, sample_metadata):
    clusterer = AudioClusterer(method='kmeans', n_clusters=3)
    clusterer.fit(sample_embeddings)
    
    analysis = analyze_clusters(sample_embeddings, clusterer.labels, sample_metadata)
    
    assert len(analysis) == 3
    for cluster_info in analysis.values():
        assert 'size' in cluster_info
        assert 'genres' in cluster_info