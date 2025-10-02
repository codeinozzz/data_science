import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.ingestion.audio_loader import AudioLoader
from src.ingestion.audio_processor import AudioProcessor
from src.embeddings.audio_embedder import AudioEmbedder
from src.storage.chroma_client import ChromaStorage


def main():
    data_path = Path(__file__).parent.parent / "data" / "raw"
    db_path = Path(__file__).parent.parent / "chroma_db"

    loader = AudioLoader(str(data_path))
    processor = AudioProcessor()
    embedder = AudioEmbedder()
    storage = ChromaStorage(persist_directory=str(db_path))

    print(f"Database has {storage.count()} samples\n")

    print("Loading a query sample...")
    all_audio = loader.load_all()
    query_audio = all_audio[0]

    print(f"Query: {query_audio['filename']} (genre: {query_audio['genre']})")

    processed = processor.extract_features(query_audio)
    query_embedding = embedder.generate_embedding(processed)

    print("\nSearching for similar samples...")
    results = storage.search_similar(query_embedding, n_results=5)

    print("\nTop 5 similar samples:")
    for i, (metadata, distance) in enumerate(
        zip(results["metadatas"][0], results["distances"][0]), 1
    ):
        print(f"{i}. {metadata['filename']}")
        print(f"   Genre: {metadata['genre']}")
        print(f"   Distance: {distance:.4f}\n")


if __name__ == "__main__":
    main()
