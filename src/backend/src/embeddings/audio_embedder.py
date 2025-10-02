import numpy as np
from typing import List, Dict


class AudioEmbedder:
    def __init__(self):
        self.feature_dim = None

    def generate_embedding(self, processed_audio: Dict) -> np.ndarray:
        feature_vector = processed_audio["features"]

        if self.feature_dim is None:
            self.feature_dim = len(feature_vector)

        normalized = (feature_vector - np.mean(feature_vector)) / (
            np.std(feature_vector) + 1e-8
        )

        return normalized

    def generate_embeddings_batch(
        self, processed_batch: List[Dict]
    ) -> List[np.ndarray]:
        embeddings = []
        for processed in processed_batch:
            embedding = self.generate_embedding(processed)
            embeddings.append(embedding)
        return embeddings

    def get_feature_dimension(self) -> int:
        return self.feature_dim
