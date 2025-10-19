import numpy as np
from sklearn.manifold import TSNE
import joblib
from pathlib import Path

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


class DimensionalityReducer:
    def __init__(self, method='tsne', n_components=2):
        self.method = method
        self.n_components = n_components
        self.model = None
        self.embedding = None
        
    def fit_transform(self, data):
        """Reduce dimensionality of data"""
        if self.method == 'tsne':
            self.model = TSNE(
                n_components=self.n_components,
                random_state=42,
                perplexity=30,
                n_iter=1000
            )
            self.embedding = self.model.fit_transform(data)
            
        elif self.method == 'umap':
            if not UMAP_AVAILABLE:
                raise ImportError("umap-learn not installed. Run: pip install umap-learn")
            
            self.model = umap.UMAP(
                n_components=self.n_components,
                random_state=42,
                n_neighbors=15,
                min_dist=0.1
            )
            self.embedding = self.model.fit_transform(data)
            
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return self.embedding
    
    def save(self, filepath):
        """Save reduced embeddings to disk"""
        np.save(filepath, self.embedding)
        
    @classmethod
    def load(cls, filepath):
        """Load reduced embeddings from disk"""
        reducer = cls()
        reducer.embedding = np.load(filepath)
        return reducer


def reduce_to_2d(embeddings, method='tsne'):
    """Helper function to reduce embeddings to 2D"""
    reducer = DimensionalityReducer(method=method, n_components=2)
    return reducer.fit_transform(embeddings)


def reduce_to_3d(embeddings, method='tsne'):
    """Helper function to reduce embeddings to 3D"""
    reducer = DimensionalityReducer(method=method, n_components=3)
    return reducer.fit_transform(embeddings)