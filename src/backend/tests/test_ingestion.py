import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.ingestion.audio_loader import AudioLoader
from src.ingestion.audio_processor import AudioProcessor


def main():
    data_path = Path(__file__).parent.parent / "data" / "raw"

    print(f"Data directory: {data_path.absolute()}")

    if not data_path.exists():
        print("ERROR: Directory not found")
        return

    loader = AudioLoader(str(data_path))
    processor = AudioProcessor()

    print("Loading samples...")
    all_audio = loader.load_all()
    print(f"Loaded: {len(all_audio)} audio files")

    if len(all_audio) == 0:
        print("No audio files found")
        return

    genres = {}
    for audio in all_audio:
        genre = audio["genre"]
        genres[genre] = genres.get(genre, 0) + 1

    print("\nSamples per genre:")
    for genre, count in genres.items():
        print(f"  {genre}: {count}")

    print("\nProcessing first sample...")
    processed = processor.extract_features(all_audio[0])

    print(f"Feature vector shape: {processed['features'].shape}")
    print(f"Feature vector dimensions: {len(processed['features'])}")
    print(f"\nMetadata:")
    print(f"  Filename: {processed['metadata']['filename']}")
    print(f"  Genre: {processed['metadata']['genre']}")
    print(f"  Duration: {processed['metadata']['duration']:.2f}s")
    print(f"\nKey features:")
    print(f"  Tempo: {processed['feature_names']['tempo']:.1f} BPM")
    print(
        f"  Spectral centroid: {processed['feature_names']['spectral_centroid_mean']:.1f} Hz"
    )


if __name__ == "__main__":
    main()
