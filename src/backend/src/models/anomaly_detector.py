from sklearn.ensemble import IsolationForest
import numpy as np
from typing import List, Tuple


class AnomalyDetector:
    def __init__(self, contamination: float = 0.1):
        self.model = IsolationForest(
            contamination=contamination, random_state=42, n_estimators=100
        )
        self.is_fitted = False

    def fit(self, embeddings: np.ndarray):
        self.model.fit(embeddings)
        self.is_fitted = True

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        predictions = self.model.predict(embeddings)
        return predictions

    def get_anomalies(
        self, embeddings: np.ndarray, metadata: List[dict]
    ) -> List[Tuple[dict, float]]:
        predictions = self.predict(embeddings)
        scores = self.model.score_samples(embeddings)

        anomalies = []
        for i, (pred, score) in enumerate(zip(predictions, scores)):
            if pred == -1:
                anomalies.append((metadata[i], float(score)))

        anomalies.sort(key=lambda x: x[1])
        return anomalies
