import sys
from pathlib import Path
import numpy as np
import json

sys.path.append(str(Path(__file__).parent.parent))

from src.storage.chroma_client import ChromaStorage
from src.models.clustering import AudioClusterer, find_optimal_k, analyze_clusters
from src.models.dimensionality_reduction import DimensionalityReducer
from src.models.anomaly_detector import AnomalyDetector
from config.settings import (
    N_CLUSTERS,
    CLUSTERING_METHOD,
    REDUCTION_METHOD,
    PROCESSED_DIR,
    MODELS_DIR,
    CHROMA_DB_DIR,
)


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    storage = ChromaStorage(persist_directory=str(CHROMA_DB_DIR))
    all_data = storage.get_all_samples()
    embeddings = np.array(all_data["embeddings"])
    metadata = all_data["metadatas"]

    k_analysis = find_optimal_k(embeddings, k_range=(2, 10))
    best_k_idx = np.argmax(k_analysis["silhouette_scores"])
    best_k = k_analysis["k_values"][best_k_idx]

    clusterer = AudioClusterer(method=CLUSTERING_METHOD, n_clusters=N_CLUSTERS)
    clusterer.fit(embeddings)

    reducer_2d = DimensionalityReducer(method=REDUCTION_METHOD, n_components=2)
    embeddings_2d = reducer_2d.fit_transform(embeddings)

    reducer_3d = DimensionalityReducer(method=REDUCTION_METHOD, n_components=3)
    embeddings_3d = reducer_3d.fit_transform(embeddings)

    centroids_2d = None
    centroid_info = []

    if clusterer.method == "kmeans":
        centroids_2d = []

        for cluster_id in range(clusterer.n_clusters):
            mask = clusterer.labels == cluster_id
            cluster_points = embeddings_2d[mask]

            if len(cluster_points) > 0:
                centroid = cluster_points.mean(axis=0)
                centroids_2d.append(centroid)

                radius = np.mean(np.linalg.norm(cluster_points - centroid, axis=1))

                centroid_info.append(
                    {
                        "cluster_id": int(cluster_id),
                        "centroid_x": float(centroid[0]),
                        "centroid_y": float(centroid[1]),
                        "n_samples": int(np.sum(mask)),
                        "avg_radius": float(radius),
                    }
                )

        centroids_2d = np.array(centroids_2d)

    detector = AnomalyDetector(contamination=0.1)
    detector.fit(embeddings)

    predictions = detector.predict(embeddings)
    scores = detector.model.score_samples(embeddings)

    anomaly_indices = np.where(predictions == -1)[0]

    model_path = MODELS_DIR / f"clusterer_{CLUSTERING_METHOD}.pkl"
    clusterer.save(model_path)

    np.save(PROCESSED_DIR / f"embeddings_2d_{REDUCTION_METHOD}.npy", embeddings_2d)
    np.save(PROCESSED_DIR / f"embeddings_3d_{REDUCTION_METHOD}.npy", embeddings_3d)

    np.save(PROCESSED_DIR / "cluster_labels.npy", clusterer.labels)

    if centroids_2d is not None:
        np.save(PROCESSED_DIR / "centroids_2d.npy", centroids_2d)
        with open(PROCESSED_DIR / "centroids_info.json", "w") as f:
            json.dump(centroid_info, f, indent=2)

    clusters_info = analyze_clusters(embeddings, clusterer.labels, metadata)
    with open(PROCESSED_DIR / "clusters_analysis.json", "w") as f:
        json.dump(clusters_info, f, indent=2)

    with open(PROCESSED_DIR / "k_analysis.json", "w") as f:
        json.dump(k_analysis, f, indent=2)

    metadata_enhanced = []
    for i, meta in enumerate(metadata):
        meta_copy = meta.copy()
        meta_copy["cluster"] = int(clusterer.labels[i])
        meta_copy["embedding_2d"] = embeddings_2d[i].tolist()
        meta_copy["embedding_3d"] = embeddings_3d[i].tolist()
        meta_copy["is_anomaly"] = bool(predictions[i] == -1)
        meta_copy["anomaly_score"] = float(scores[i])
        metadata_enhanced.append(meta_copy)

    with open(PROCESSED_DIR / "metadata_with_clusters.json", "w") as f:
        json.dump(metadata_enhanced, f, indent=2)

    with open(PROCESSED_DIR / "metadata_with_anomalies.json", "w") as f:
        json.dump(metadata_enhanced, f, indent=2)


if __name__ == "__main__":
    main()
