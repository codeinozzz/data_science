from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import tempfile
import sys
import logging
import json
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.ingestion.audio_loader import AudioLoader
from src.ingestion.audio_processor import AudioProcessor
from src.embeddings.audio_embedder import AudioEmbedder
from src.storage.chroma_client import ChromaStorage
from src.models.clustering import AudioClusterer
from src.api.models import (
    SearchResponse,
    GenreStats,
    HealthResponse,
    QueryInfo,
    SearchResult,
)
from utils.validators import validate_audio_file, validate_search_params
from config.settings import CHROMA_DB_DIR, PROCESSED_DIR, MODELS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Audio Samples Semantic Search API",
    description="Sistema de búsqueda semántica para samples de música electrónica",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

storage = ChromaStorage(persist_directory=str(CHROMA_DB_DIR))
processor = AudioProcessor()
embedder = AudioEmbedder()

MAX_FILE_SIZE = 10 * 1024 * 1024


@app.get("/", response_model=HealthResponse)
def root():
    """API health check and available endpoints"""
    return HealthResponse(
        status="online",
        message="Audio Samples Semantic Search API v2.0",
        endpoints=[
            "/stats",
            "/search",
            "/clusters",
            "/search-by-filters",
            "/ingest",
            "/docs",
        ],
    )


@app.get("/stats", response_model=GenreStats)
def get_stats():
    """Get dataset statistics"""
    try:
        all_data = storage.get_all_samples()
        genres = {}

        for meta in all_data["metadatas"]:
            genre = meta["genre"]
            genres[genre] = genres.get(genre, 0) + 1

        return GenreStats(
            total_samples=storage.count(), genres=genres, embedding_dimension=45
        )
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(500, "Error retrieving statistics")


@app.post("/search", response_model=SearchResponse)
async def search_similar(
    file: UploadFile = File(..., description="Audio file (MP3/WAV/FLAC, max 10MB)"),
    n_results: int = Query(5, ge=1, le=20, description="Number of results to return"),
):
    """Search for similar audio samples using semantic similarity"""
    validate_search_params(n_results)

    content = await file.read()
    file_size = len(content)

    validate_audio_file(file.filename, file_size)

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=Path(file.filename).suffix
    ) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        loader = AudioLoader(str(tmp_path.parent))
        audio_data = loader.load_audio(tmp_path)

        if not audio_data:
            raise HTTPException(400, "Failed to load audio file")

        processed = processor.extract_features(audio_data)
        query_embedding = embedder.generate_embedding(processed)

        results = storage.search_similar(query_embedding, n_results)

        search_results = [
            SearchResult(
                filename=meta["filename"], genre=meta["genre"], distance=float(dist)
            )
            for meta, dist in zip(results["metadatas"][0], results["distances"][0])
        ]

        logger.info(
            f"Search completed: {file.filename}, found {len(search_results)} results"
        )

        return SearchResponse(
            query=QueryInfo(filename=file.filename, duration=audio_data["duration"]),
            results=search_results,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing search: {e}")
        raise HTTPException(500, "Error processing audio file")
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


@app.get("/clusters")
def get_clusters():
    """
    Get clustering information and analysis.
    Returns cluster composition, sizes, and genre distribution.
    """
    try:
        clusters_file = PROCESSED_DIR / "clusters_analysis.json"

        if not clusters_file.exists():
            raise HTTPException(
                404, "Clustering data not found. Run scripts/analyze_clusters.py first"
            )

        with open(clusters_file, "r") as f:
            clusters_info = json.load(f)

        return {"total_clusters": len(clusters_info), "clusters": clusters_info}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading clusters: {e}")
        raise HTTPException(500, "Error loading cluster data")


@app.get("/search-by-filters")
def search_by_filters(
    genre: str = Query(None, description="Filter by genre"),
    cluster: int = Query(None, description="Filter by cluster ID"),
    limit: int = Query(20, ge=1, le=100, description="Max results"),
):
    """
    Search samples using filters (genre, cluster).
    Returns samples matching the criteria.
    """
    try:
        metadata_file = PROCESSED_DIR / "metadata_with_clusters.json"

        if not metadata_file.exists():
            raise HTTPException(
                404, "Metadata not found. Run scripts/analyze_clusters.py first"
            )

        with open(metadata_file, "r") as f:
            all_metadata = json.load(f)

        filtered = all_metadata

        if genre:
            filtered = [m for m in filtered if m.get("genre") == genre]

        if cluster is not None:
            filtered = [m for m in filtered if m.get("cluster") == cluster]

        filtered = filtered[:limit]

        return {
            "total_results": len(filtered),
            "filters_applied": {"genre": genre, "cluster": cluster, "limit": limit},
            "results": filtered,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in filtered search: {e}")
        raise HTTPException(500, "Error processing filtered search")


@app.post("/ingest")
async def ingest_sample(
    file: UploadFile = File(..., description="Audio file to add to database"),
    genre: str = Query(..., description="Genre label for the sample"),
):
    """
    Ingest a new audio sample into the database.
    Processes the audio, generates embedding, assigns cluster, and stores in ChromaDB.
    """
    content = await file.read()
    file_size = len(content)

    validate_audio_file(file.filename, file_size)

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=Path(file.filename).suffix
    ) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        loader = AudioLoader(str(tmp_path.parent))
        audio_data = loader.load_audio(tmp_path)

        if not audio_data:
            raise HTTPException(400, "Failed to load audio file")

        audio_data["genre"] = genre

        processed = processor.extract_features(audio_data)
        embedding = embedder.generate_embedding(processed)

        model_path = MODELS_DIR / "clusterer_kmeans.pkl"
        if model_path.exists():
            clusterer = AudioClusterer.load(model_path)
            cluster_id = int(clusterer.predict([embedding])[0])
        else:
            cluster_id = -1

        sample_id = f"sample_{storage.count() + 1}"

        metadata = processed["metadata"]
        metadata["cluster"] = cluster_id

        storage.add_samples(
            embeddings=[embedding], metadata=[metadata], ids=[sample_id]
        )

        logger.info(f"Ingested new sample: {file.filename} (cluster: {cluster_id})")

        return {
            "status": "success",
            "message": f"Sample {file.filename} added to database",
            "sample_id": sample_id,
            "cluster": cluster_id,
            "genre": genre,
            "total_samples": storage.count(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ingesting sample: {e}")
        raise HTTPException(500, f"Error ingesting sample: {str(e)}")
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
