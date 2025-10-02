from pathlib import Path
from typing import List, Dict, Optional
import librosa
import soundfile as sf
import numpy as np


class AudioLoader:
    def __init__(self, data_dir: str, sample_rate: int = 22050):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate

    def load_audio(self, file_path: Path) -> Optional[Dict]:
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            duration = librosa.get_duration(y=audio, sr=sr)

            return {
                "audio": audio,
                "sample_rate": sr,
                "duration": duration,
                "filename": file_path.name,
                "path": str(file_path),
                "genre": file_path.parent.name,
            }
        except Exception as e:
            return None

    def load_genre(self, genre: str) -> List[Dict]:
        genre_path = self.data_dir / genre
        if not genre_path.exists():
            return []

        audio_files = []
        for ext in ["*.mp3", "*.wav", "*.flac"]:
            audio_files.extend(list(genre_path.glob(ext)))

        loaded = []
        for file_path in audio_files:
            audio_data = self.load_audio(file_path)
            if audio_data:
                loaded.append(audio_data)

        return loaded

    def load_all(self) -> List[Dict]:
        all_audio = []

        for genre_dir in self.data_dir.iterdir():
            if genre_dir.is_dir():
                audio_list = self.load_genre(genre_dir.name)
                all_audio.extend(audio_list)

        return all_audio
