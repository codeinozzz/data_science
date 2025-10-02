import librosa
import numpy as np
from typing import Dict


class AudioProcessor:
    def __init__(self, n_mfcc: int = 13, n_chroma: int = 12):
        self.n_mfcc = n_mfcc
        self.n_chroma = n_chroma

    def extract_features(self, audio_data: Dict) -> Dict:
        audio = audio_data["audio"]
        sr = audio_data["sample_rate"]

        features = {}

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc)
        features["mfcc_mean"] = np.mean(mfcc, axis=1)
        features["mfcc_std"] = np.std(mfcc, axis=1)

        chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_chroma=self.n_chroma)
        features["chroma_mean"] = np.mean(chroma, axis=1)

        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        features["spectral_centroid_mean"] = np.mean(spectral_centroid)
        features["spectral_centroid_std"] = np.std(spectral_centroid)

        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        features["spectral_rolloff_mean"] = np.mean(spectral_rolloff)

        zcr = librosa.feature.zero_crossing_rate(audio)
        features["zcr_mean"] = np.mean(zcr)

        try:
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
            features["tempo"] = float(tempo)
        except:
            features["tempo"] = 0.0

        rms = librosa.feature.rms(y=audio)
        features["rms_mean"] = np.mean(rms)

        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        features["spectral_bandwidth_mean"] = np.mean(spectral_bandwidth)

        feature_vector = np.concatenate(
            [
                features["mfcc_mean"],
                features["mfcc_std"],
                features["chroma_mean"],
                [features["spectral_centroid_mean"]],
                [features["spectral_centroid_std"]],
                [features["spectral_rolloff_mean"]],
                [features["zcr_mean"]],
                [features["tempo"]],
                [features["rms_mean"]],
                [features["spectral_bandwidth_mean"]],
            ]
        )

        return {
            "features": feature_vector,
            "feature_names": features,
            "metadata": {
                "filename": audio_data["filename"],
                "genre": audio_data["genre"],
                "duration": audio_data["duration"],
                "path": audio_data["path"],
            },
        }

    def process_batch(self, audio_list: list) -> list:
        processed = []
        for idx, audio_data in enumerate(audio_list):
            try:
                result = self.extract_features(audio_data)
                processed.append(result)
            except Exception as e:
                print(f"Error processing {audio_data['filename']}: {e}")
        return processed
