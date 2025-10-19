import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
import joblib
from pathlib import Path


class AudioClusterer:
    def __init__(self, method='kmeans', n_clusters=6):
        self.method = method
        self.n_clusters = n_clusters
        self.model = None
        self.labels = None
        self.metrics = {}
        
    def fit(self, embeddings):
        """Fit clustering model on embeddings"""
        if self.method == 'kmeans':
            self.model = KMeans(n_clusters=self.n_clusters, random_state=42)
        elif self.method == 'dbscan':
            self.model = DBSCAN(eps=0.5, min_samples=5)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.labels = self.model.fit_predict(embeddings)
        self._calculate_metrics(embeddings)
        return self
    
    def _calculate_metrics(self, embeddings):
        """Calculate clustering quality metrics"""
        if len(np.unique(self.labels)) > 1:
            self.metrics['silhouette'] = silhouette_score(embeddings, self.labels)
            self.metrics['davies_bouldin'] = davies_bouldin_score(embeddings, self.labels)
        self.metrics['n_clusters'] = len(np.unique(self.labels))
        
    def predict(self, embeddings):
        """Predict cluster for new embeddings"""
        if self.method == 'kmeans':
            return self.model.predict(embeddings)
        elif self.method == 'dbscan':
            raise NotImplementedError("DBSCAN doesn't support predict")
            
    def save(self, filepath):
        """Save model to disk"""
        joblib.dump({
            'model': self.model,
            'labels': self.labels,
            'metrics': self.metrics,
            'method': self.method,
            'n_clusters': self.n_clusters
        }, filepath)
        
    @classmethod
    def load(cls, filepath):
        """Load model from disk"""
        data = joblib.load(filepath)
        clusterer = cls(method=data['method'], n_clusters=data['n_clusters'])
        clusterer.model = data['model']
        clusterer.labels = data['labels']
        clusterer.metrics = data['metrics']
        return clusterer


def find_optimal_k(embeddings, k_range=(2, 10)):
    """Find optimal number of clusters using elbow method and silhouette"""
    silhouette_scores = []
    inertias = []
    
    for k in range(k_range[0], k_range[1] + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        
        silhouette_scores.append(silhouette_score(embeddings, labels))
        inertias.append(kmeans.inertia_)
    
    return {
        'k_values': list(range(k_range[0], k_range[1] + 1)),
        'silhouette_scores': silhouette_scores,
        'inertias': inertias
    }


def analyze_clusters(embeddings, labels, metadata):
    """Analyze cluster composition"""
    clusters_info = {}
    
    for cluster_id in np.unique(labels):
        mask = labels == cluster_id
        cluster_metadata = [m for m, is_in_cluster in zip(metadata, mask) if is_in_cluster]
        
        genres = {}
        for meta in cluster_metadata:
            genre = meta.get('genre', 'unknown')
            genres[genre] = genres.get(genre, 0) + 1
        
        clusters_info[int(cluster_id)] = {
            'size': int(np.sum(mask)),
            'genres': genres,
            'samples': [m['filename'] for m in cluster_metadata[:5]]
        }
    
    return clusters_info