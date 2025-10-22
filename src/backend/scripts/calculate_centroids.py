import sys
from pathlib import Path
import numpy as np
import json

sys.path.append(str(Path(__file__).parent.parent))

from config.settings import PROCESSED_DIR, MODELS_DIR
from src.models.clustering import AudioClusterer


def main():
    embeddings_2d = np.load(PROCESSED_DIR / "embeddings_2d_tsne.npy")
    labels = np.load(PROCESSED_DIR / "cluster_labels.npy")

    model_path = MODELS_DIR / "clusterer_kmeans.pkl"

    if not model_path.exists():
        return

    clusterer = AudioClusterer.load(model_path)

    if clusterer.method != "kmeans":
        return

    centroids_2d = []
    centroid_info = []

    for cluster_id in range(clusterer.n_clusters):
        mask = labels == cluster_id
        cluster_points = embeddings_2d[mask]

        if len(cluster_points) > 0:
            centroid = cluster_points.mean(axis=0)
            centroids_2d.append(centroid)

            std_dev = cluster_points.std(axis=0)
            radius = np.mean(np.linalg.norm(cluster_points - centroid, axis=1))

            centroid_info.append(
                {
                    "cluster_id": int(cluster_id),
                    "centroid_x": float(centroid[0]),
                    "centroid_y": float(centroid[1]),
                    "n_samples": int(np.sum(mask)),
                    "std_x": float(std_dev[0]),
                    "std_y": float(std_dev[1]),
                    "avg_radius": float(radius),
                }
            )

    centroids_2d = np.array(centroids_2d)

    output_npy = PROCESSED_DIR / "centroids_2d.npy"
    np.save(output_npy, centroids_2d)

    output_json = PROCESSED_DIR / "centroids_info.json"
    with open(output_json, "w") as f:
        json.dump(centroid_info, f, indent=2)


if __name__ == "__main__":
    main()
