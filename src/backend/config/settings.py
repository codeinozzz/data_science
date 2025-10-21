import os
from pathlib import Path

# API
FREESOUND_API_KEY = os.getenv("FREESOUND_API_KEY", "")

# ChromaDB
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
CHROMA_COLLECTION = "audio_samples"

# Backend
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8001"))
MAX_UPLOAD_SIZE_MB = 10
ALLOWED_EXTENSIONS = [".mp3", ".wav", ".flac"]

# Audio Processing
SAMPLE_RATE = 22050
N_MFCC = 13
N_CHROMA = 12

# ML Models
ANOMALY_CONTAMINATION = 0.1
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 20
RANDOM_STATE = 42

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"


# Clustering Configuration
N_CLUSTERS = 6
CLUSTERING_METHOD = "kmeans"
DBSCAN_EPS = 0.5
DBSCAN_MIN_SAMPLES = 5

# Dimensionality Reduction
REDUCTION_METHOD = "tsne"
TSNE_PERPLEXITY = 30
TSNE_N_ITER = 1000
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
N_COMPONENTS_2D = 2
N_COMPONENTS_3D = 3

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models" / "saved"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"
