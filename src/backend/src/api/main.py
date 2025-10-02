from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import tempfile
import numpy as np
from typing import List, Dict

import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.ingestion.audio_loader import AudioLoader
from src.ingestion.audio_processor import AudioProcessor
from src.embeddings.audio_embedder import AudioEmbedder
from src.storage.chroma_client import ChromaStorage

app = FastAPI(title="Music Samples API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db_path = Path(__file__).parent.parent.parent / "chroma_db"
storage = ChromaStorage(persist_directory=str(db_path))
processor = AudioProcessor()
embedder = AudioEmbedder()


@app.get("/")
def root():
    return {
        "message": "Music Samples Capstone API",
        "endpoints": ["/stats", "/search", "/upload"],
    }


@app.get("/stats")
def get_stats():
    all_data = storage.get_all_samples()
    genres = {}

    for meta in all_data["metadatas"]:
        genre = meta["genre"]
        genres[genre] = genres.get(genre, 0) + 1

    return {
        "total_samples": storage.count(),
        "genres": genres,
        "embedding_dimension": 45,
    }


@app.post("/search")
async def search_similar(file: UploadFile = File(...), n_results: int = 5):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        loader = AudioLoader(str(tmp_path.parent))
        audio_data = loader.load_audio(tmp_path)

        if not audio_data:
            return {"error": "Failed to load audio"}

        processed = processor.extract_features(audio_data)
        query_embedding = embedder.generate_embedding(processed)

        results = storage.search_similar(query_embedding, n_results)

        return {
            "query": {"filename": file.filename, "duration": audio_data["duration"]},
            "results": [
                {
                    "filename": meta["filename"],
                    "genre": meta["genre"],
                    "distance": float(dist),
                }
                for meta, dist in zip(results["metadatas"][0], results["distances"][0])
            ],
        }
    finally:
        tmp_path.unlink()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
