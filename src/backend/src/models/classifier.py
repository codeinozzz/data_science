from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from typing import Tuple, Dict


class GenreClassifier:
    def __init__(self, n_estimators: int = 100):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, random_state=42, max_depth=20
        )
        self.is_fitted = False
        self.classes = None

    def prepare_data(
        self, embeddings: np.ndarray, labels: list
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels, test_size=0.2, random_state=42, stratify=labels
        )
        return X_train, X_test, y_train, y_test

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(X_train, y_train)
        self.classes = self.model.classes_
        self.is_fitted = True

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        predictions = self.model.predict(X_test)

        report = classification_report(y_test, predictions, output_dict=True)

        conf_matrix = confusion_matrix(y_test, predictions)

        return {
            "accuracy": report["accuracy"],
            "report": report,
            "confusion_matrix": conf_matrix.tolist(),
            "classes": self.classes.tolist(),
        }

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return self.model.predict(embeddings)

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return self.model.predict_proba(embeddings)
