## Overview

This project implements an automated system to organize, search, and classify electronic music samples using audio feature extraction, vector embeddings, and unsupervised learning techniques.

**Dataset:** 267 audio samples across 6 genres (ambient, drum & bass, dubstep, house, techno, trance)

## Features

- Audio feature extraction (MFCCs, spectral, temporal features)
- 45-dimensional embeddings generation
- Semantic similarity search
- Anomaly detection (identifies 27 anomalous samples)
- Genre classification (33% accuracy with RandomForest)
- REST API for search and analytics
- Dockerized deployment with ChromaDB

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Docker Environment                    │
│                                                          │
│  ┌──────────────────┐         ┌─────────────────────┐  │
│  │   ChromaDB       │         │   Backend API       │  │
│  │   Container      │◄────────┤   Container         │  │
│  │   Port: 8000     │         │   Port: 8001        │  │
│  └──────────────────┘         └─────────────────────┘  │
│          │                              │               │
│  ┌───────▼──────────────────────────────▼──────────┐   │
│  │            Persistent Volumes                    │   │
│  │  • chroma_data (vector database)                │   │
│  │  • backend/data (audio samples)                 │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Data Pipeline

```
Audio Files (MP3/WAV)
        ↓
┌───────────────────────────────────┐
│  Ingestion Module                 │
│  • AudioLoader (librosa)          │
│  • AudioProcessor (feature ext.)  │
└───────────────┬───────────────────┘
                ↓
        Feature Vector (45-dim)
        • 13 MFCCs (mean)
        • 13 MFCCs (std)
        • 12 Chroma features
        • Spectral centroid
        • Spectral rolloff
        • Zero crossing rate
        • Tempo (BPM)
        • RMS energy
        • Spectral bandwidth
                ↓
┌───────────────────────────────────┐
│  Embeddings Module                │
│  • Normalization                  │
│  • AudioEmbedder                  │
└───────────────┬───────────────────┘
                ↓
        Normalized Embeddings
                ↓
┌───────────────────────────────────┐
│  ChromaDB Storage                 │
│  • Vector similarity index        │
│  • Metadata storage               │
└───────────────┬───────────────────┘
                ↓
        ┌───────┴───────┐
        ↓               ↓
┌───────────────┐ ┌────────────────┐
│  Search API   │ │  ML Models     │
│  • Semantic   │ │  • Anomaly Det │
│  • Stats      │ │  • Classifier  │
└───────────────┘ └────────────────┘
```

### Component Details

**1. Ingestion Layer**

- `audio_loader.py`: Loads MP3/WAV files with librosa
- `audio_processor.py`: Extracts 45 audio features (MFCCs, spectral, temporal)

**2. Embeddings Layer**

- `audio_embedder.py`: Normalizes feature vectors to create embeddings

**3. Storage Layer**

- `chroma_client.py`: ChromaDB HTTP client for vector storage and similarity search

**4. Models Layer**

- `anomaly_detector.py`: Isolation Forest for anomaly detection
- `classifier.py`: Random Forest for genre classification

**5. API Layer**

- `main.py`: FastAPI endpoints for search, stats, and file upload

## Tech Stack

- **Backend:** Python 3.10, FastAPI, Uvicorn
- **ML/Audio:** librosa 0.10.1, scikit-learn 1.4.0, numpy 1.24.3
- **Vector DB:** ChromaDB 0.4.15
- **Deployment:** Docker, Docker Compose

## Quick Start

bash

```bash
# Clone repository
cd your-project-directory

# Build and start services
docker-compose up -d

# Build the vector database (first time only)
docker-compose exec backend python scripts/build_database.py

# Verify system is running
curl http://localhost:8001/stats
```

Expected output:

json

```json
{
  "total_samples": 267,
  "genres": {
    "ambient": 47,
    "drum_and_bass": 50,
    "dubstep": 34,
    "techno": 44,
    "house": 50,
    "trance": 42
  },
  "embedding_dimension": 45
}
```

## API Endpoints

### GET /

Returns API information and available endpoints.

### GET /stats

Returns dataset statistics including total samples, genre distribution, and embedding dimensions.

### POST /search

Performs semantic similarity search. Upload an audio file to find similar samples.

**Example:**

bash

```bash
curl -X POST -F "file=@sample.mp3" http://localhost:8001/search?n_results=5
```

## Project Structure

```
backend/
├── src/
│   ├── ingestion/          # Audio loading and processing
│   │   ├── audio_loader.py
│   │   └── audio_processor.py
│   ├── embeddings/         # Feature extraction and normalization
│   │   └── audio_embedder.py
│   ├── storage/            # ChromaDB integration
│   │   └── chroma_client.py
│   ├── models/             # ML models
│   │   ├── anomaly_detector.py
│   │   └── classifier.py
│   └── api/                # FastAPI endpoints
│       └── main.py
├── scripts/                # Utilities and evaluation
│   ├── download_freesound.py
│   ├── build_database.py
│   ├── evaluate_system.py
│   └── test_search.py
└── data/                   # Audio samples (not in repo)
    └── raw/
```

## System Metrics

- **Total samples:** 267
- **Embedding dimension:** 45
- **Anomalies detected:** 27 (10%)
- **Classification accuracy:** 33%
- **Database:** ChromaDB with persistent storage

### Performance Breakdown

**Anomaly Detection:**

- Method: Isolation Forest
- Contamination rate: 10%
- Top anomalous genres: ambient (6), drum_and_bass (3)

**Genre Classification:**

- Model: Random Forest (100 estimators)
- Train/Test split: 80/20
- Best performing: Trance (66% precision), Ambient (60% recall)
- Worst performing: Dubstep (0% precision)

## Requirements

- Docker & Docker Compose
- 512MB RAM minimum
- 100MB disk space

## Development

### Run System Evaluation

bash

```bash
docker-compose exec backend python scripts/evaluate_system.py
```

Output includes:

- Anomaly detection results
- Classification accuracy per genre
- Confusion matrix

### Test Semantic Search

bash

```bash
docker-compose exec backend python scripts/test_search.py
```

### Download More Samples

bash

```bash
# Set FREESOUND_API_KEY in backend/.env
docker-compose exec backend python scripts/download_freesound.py
```

## Troubleshooting

**ChromaDB connection issues:**

bash

```bash
# Check if services are running
docker-compose ps

# View logs
docker-compose logs chromadb
docker-compose logs backend

# Restart services
docker-compose restart
```

**Rebuild database:**

bash

```bash
docker-compose exec backend python scripts/build_database.py
```
