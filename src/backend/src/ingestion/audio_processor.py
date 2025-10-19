import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.feature_extractors import (
    extract_mfcc,
    extract_chroma,
    extract_spectral_centroid,
    extract_spectral_rolloff,
    extract_zero_crossing_rate,
    extract_tempo,
    extract_rms,
    extract_spectral_bandwidth
)


class AudioProcessor:
    def __init__(self, n_mfcc=13, n_chroma=12):
        self.n_mfcc = n_mfcc
        self.n_chroma = n_chroma

    def extract_features(self, audio_data):
        audio = audio_data["audio"]
        sr = audio_data["sample_rate"]

        mfcc_mean, mfcc_std = extract_mfcc(audio, sr, self.n_mfcc)
        chroma_mean = extract_chroma(audio, sr, self.n_chroma)
        centroid_mean, centroid_std = extract_spectral_centroid(audio, sr)
        rolloff_mean = extract_spectral_rolloff(audio, sr)
        zcr_mean = extract_zero_crossing_rate(audio)
        tempo = extract_tempo(audio, sr)
        rms_mean = extract_rms(audio)
        bandwidth_mean = extract_spectral_bandwidth(audio, sr)

        feature_vector = np.concatenate([
            mfcc_mean,
            mfcc_std,
            chroma_mean,
            [centroid_mean],
            [centroid_std],
            [rolloff_mean],
            [zcr_mean],
            [tempo],
            [rms_mean],
            [bandwidth_mean]
        ])

        return {
            "features": feature_vector,
            "metadata": {
                "filename": audio_data["filename"],
                "genre": audio_data["genre"],
                "duration": audio_data["duration"],
                "path": audio_data["path"]
            }
        }

    def process_batch(self, audio_list):
        processed = []
        for audio_data in audio_list:
            try:
                result = self.extract_features(audio_data)
                processed.append(result)
            except Exception as e:
                print(f"Error processing {audio_data['filename']}: {e}")
        return processed