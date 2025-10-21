import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from src.storage.chroma_client import ChromaStorage
from src.models.anomaly_detector import AnomalyDetector
from src.models.classifier import GenreClassifier


def main():
    db_path = Path(__file__).parent.parent / "chroma_db"
    storage = ChromaStorage(persist_directory=str(db_path))

    print("Loading data from ChromaDB...")
    all_data = storage.get_all_samples()

    if not all_data["embeddings"]:
        print("No data found in ChromaDB")
        return

    embeddings = np.array(all_data["embeddings"])
    metadata = all_data["metadatas"]
    genres = [m["genre"] for m in metadata]

    print(f"Loaded {len(embeddings)} samples")
    print(f"Embedding dimension: {embeddings.shape[1]}")

    print("\nANOMALY DETECTION")
    detector = AnomalyDetector(contamination=0.1)
    detector.fit(embeddings)

    anomalies = detector.get_anomalies(embeddings, metadata)
    print(f"Detected {len(anomalies)} anomalies:")
    for i, (meta, score) in enumerate(anomalies[:10], 1):
        print(f"{i}. {meta['filename']} (genre: {meta['genre']})")
        print(f"   Anomaly score: {score:.4f}")

    print("\nGENRE CLASSIFICATION")
    classifier = GenreClassifier()
    X_train, X_test, y_train, y_test = classifier.prepare_data(embeddings, genres)

    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    print("Training classifier...")
    classifier.train(X_train, y_train)

    print("Evaluating...")
    results = classifier.evaluate(X_test, y_test)

    print(f"\nOverall Accuracy: {results['accuracy']:.2%}")

    print("\nPer-genre metrics:")
    for genre in sorted(results["classes"]):
        if genre in results["report"]:
            metrics = results["report"][genre]
            print(f"  {genre}:")
            print(f"    Precision: {metrics['precision']:.2%}")
            print(f"    Recall: {metrics['recall']:.2%}")
            print(f"    F1-score: {metrics['f1-score']:.2%}")
            print(f"    Support: {int(metrics['support'])}")


if __name__ == "__main__":
    main()
