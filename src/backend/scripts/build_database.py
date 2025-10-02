import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.ingestion.audio_loader import AudioLoader
from src.ingestion.audio_processor import AudioProcessor
from src.embeddings.audio_embedder import AudioEmbedder
from src.storage.chroma_client import ChromaStorage


def main():
    print("=" * 60)
    print("BUILDING AUDIO SAMPLE DATABASE")
    print("=" * 60)

    data_path = Path(__file__).parent.parent / "data" / "raw"
    db_path = Path(__file__).parent.parent / "chroma_db"

    print(f"\nData directory: {data_path.absolute()}")
    print(f"DB directory: {db_path.absolute()}")

    loader = AudioLoader(str(data_path))
    processor = AudioProcessor()
    embedder = AudioEmbedder()
    storage = ChromaStorage(persist_directory=str(db_path))

    print("\nStep 1: Loading audio files...")
    all_audio = loader.load_all()
    print(f"Loaded: {len(all_audio)} samples")

    if len(all_audio) == 0:
        print("No audio files found")
        return

    print("\nStep 2: Extracting audio features...")
    processed = processor.process_batch(all_audio)
    print(f"Processed: {len(processed)} samples")

    print("\nStep 3: Generating embeddings...")
    embeddings = embedder.generate_embeddings_batch(processed)
    print(f"Generated: {len(embeddings)} embeddings")
    print(f"Embedding dimension: {embedder.get_feature_dimension()}")

    print("\nStep 4: Storing in ChromaDB...")
    ids = [f"sample_{i}" for i in range(len(processed))]
    metadata = [p["metadata"] for p in processed]

    storage.add_samples(embeddings, metadata, ids)
    print(f"Stored: {storage.count()} samples in database")

    print("\n" + "=" * 60)
    print("DATABASE BUILD COMPLETE")
    print("=" * 60)

    print("\nGenre distribution:")
    genres = {}
    for meta in metadata:
        genre = meta["genre"]
        genres[genre] = genres.get(genre, 0) + 1

    for genre, count in sorted(genres.items()):
        print(f"  {genre}: {count}")


if __name__ == "__main__":
    main()
