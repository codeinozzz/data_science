import sys
from pathlib import Path
import numpy as np
import json

sys.path.append(str(Path(__file__).parent.parent))

from src.storage.chroma_client import ChromaStorage
from src.models.clustering import AudioClusterer, find_optimal_k, analyze_clusters
from src.models.dimensionality_reduction import DimensionalityReducer
from config.settings import (
    N_CLUSTERS,
    CLUSTERING_METHOD,
    REDUCTION_METHOD,
    PROCESSED_DIR,
    MODELS_DIR,
    CHROMA_DB_DIR,
)


def main():
    print("=" * 60)
    print("AUDIO SAMPLES CLUSTERING ANALYSIS")
    print("=" * 60)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("\nStep 1: Loading embeddings from ChromaDB...")
    storage = ChromaStorage(persist_directory=str(CHROMA_DB_DIR))

    all_data = storage.get_all_samples()
    embeddings = np.array(all_data["embeddings"])
    metadata = all_data["metadatas"]

    print(f"Loaded: {len(embeddings)} samples")
    print(f"Embedding dimension: {embeddings.shape[1]}")

    print("\nStep 2: Finding optimal number of clusters...")
    k_analysis = find_optimal_k(embeddings, k_range=(2, 10))

    print("\nK-value analysis:")
    for k, silhouette in zip(k_analysis["k_values"], k_analysis["silhouette_scores"]):
        print(f"  K={k}: Silhouette={silhouette:.4f}")

    best_k_idx = np.argmax(k_analysis["silhouette_scores"])
    best_k = k_analysis["k_values"][best_k_idx]
    print(
        f"\nRecommended K: {best_k} (silhouette: {k_analysis['silhouette_scores'][best_k_idx]:.4f})"
    )
    print(f"Using configured K: {N_CLUSTERS}")

    print(f"\nStep 3: Running {CLUSTERING_METHOD.upper()} clustering...")
    clusterer = AudioClusterer(method=CLUSTERING_METHOD, n_clusters=N_CLUSTERS)
    clusterer.fit(embeddings)

    print(f"\nClustering metrics:")
    print(f"  N clusters: {clusterer.metrics['n_clusters']}")
    print(f"  Silhouette score: {clusterer.metrics['silhouette']:.4f}")
    if "davies_bouldin" in clusterer.metrics:
        print(f"  Davies-Bouldin index: {clusterer.metrics['davies_bouldin']:.4f}")

    print("\nStep 4: Analyzing cluster composition...")
    clusters_info = analyze_clusters(embeddings, clusterer.labels, metadata)

    print("\nCluster details:")
    for cluster_id, info in clusters_info.items():
        print(f"\nCluster {cluster_id}: {info['size']} samples")
        print(f"  Genres: {info['genres']}")
        print(f"  Examples: {info['samples'][:3]}")

    print(f"\nStep 5: Dimensionality reduction ({REDUCTION_METHOD.upper()})...")

    reducer_2d = DimensionalityReducer(method=REDUCTION_METHOD, n_components=2)
    embeddings_2d = reducer_2d.fit_transform(embeddings)
    print(f"Reduced to 2D: {embeddings_2d.shape}")

    reducer_3d = DimensionalityReducer(method=REDUCTION_METHOD, n_components=3)
    embeddings_3d = reducer_3d.fit_transform(embeddings)
    print(f"Reduced to 3D: {embeddings_3d.shape}")

    print("\nStep 6: Saving results...")

    model_path = MODELS_DIR / f"clusterer_{CLUSTERING_METHOD}.pkl"
    clusterer.save(model_path)
    print(f"Saved clustering model: {model_path}")

    embeddings_2d_path = PROCESSED_DIR / f"embeddings_2d_{REDUCTION_METHOD}.npy"
    np.save(embeddings_2d_path, embeddings_2d)
    print(f"Saved 2D embeddings: {embeddings_2d_path}")

    embeddings_3d_path = PROCESSED_DIR / f"embeddings_3d_{REDUCTION_METHOD}.npy"
    np.save(embeddings_3d_path, embeddings_3d)
    print(f"Saved 3D embeddings: {embeddings_3d_path}")

    labels_path = PROCESSED_DIR / "cluster_labels.npy"
    np.save(labels_path, clusterer.labels)
    print(f"Saved cluster labels: {labels_path}")

    clusters_json_path = PROCESSED_DIR / "clusters_analysis.json"
    with open(clusters_json_path, "w") as f:
        json.dump(clusters_info, f, indent=2)
    print(f"Saved cluster analysis: {clusters_json_path}")

    metadata_with_clusters = []
    for i, meta in enumerate(metadata):
        meta_copy = meta.copy()
        meta_copy["cluster"] = int(clusterer.labels[i])
        meta_copy["embedding_2d"] = embeddings_2d[i].tolist()
        meta_copy["embedding_3d"] = embeddings_3d[i].tolist()
        metadata_with_clusters.append(meta_copy)

    metadata_path = PROCESSED_DIR / "metadata_with_clusters.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata_with_clusters, f, indent=2)
    print(f"Saved enhanced metadata: {metadata_path}")

    k_analysis_path = PROCESSED_DIR / "k_analysis.json"
    with open(k_analysis_path, "w") as f:
        json.dump(k_analysis, f, indent=2)
    print(f"Saved K analysis: {k_analysis_path}")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

    print("\nGenerated files:")
    print(f"  - {model_path}")
    print(f"  - {embeddings_2d_path}")
    print(f"  - {embeddings_3d_path}")
    print(f"  - {labels_path}")
    print(f"  - {clusters_json_path}")
    print(f"  - {metadata_path}")
    print(f"  - {k_analysis_path}")

    print("\nSummary:")
    print(f"  Total samples: {len(embeddings)}")
    print(f"  Clusters: {clusterer.metrics['n_clusters']}")
    print(f"  Silhouette score: {clusterer.metrics['silhouette']:.4f}")
    print(f"  Method: {CLUSTERING_METHOD}")
    print(f"  Reduction: {REDUCTION_METHOD}")


if __name__ == "__main__":
    main()
