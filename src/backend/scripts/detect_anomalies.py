import sys
from pathlib import Path
import numpy as np
import json

sys.path.append(str(Path(__file__).parent.parent))

from src.storage.chroma_client import ChromaStorage
from src.models.anomaly_detector import AnomalyDetector
from config.settings import CHROMA_DB_DIR, PROCESSED_DIR


def main():
    storage = ChromaStorage(persist_directory=str(CHROMA_DB_DIR))
    all_data = storage.get_all_samples()

    embeddings = np.array(all_data["embeddings"])
    metadata = all_data["metadatas"]

    detector = AnomalyDetector(contamination=0.1)
    detector.fit(embeddings)

    predictions = detector.predict(embeddings)
    scores = detector.model.score_samples(embeddings)

    anomaly_indices = np.where(predictions == -1)[0]

    metadata_file = PROCESSED_DIR / "metadata_with_clusters.json"
    
    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            metadata_enhanced = json.load(f)
    else:
        metadata_enhanced = metadata

    for i in range(len(metadata_enhanced)):
        metadata_enhanced[i]["is_anomaly"] = bool(predictions[i] == -1)
        metadata_enhanced[i]["anomaly_score"] = float(scores[i])

    output_path = PROCESSED_DIR / "metadata_with_anomalies.json"
    with open(output_path, "w") as f:
        json.dump(metadata_enhanced, f, indent=2)


if __name__ == "__main__":
    main()