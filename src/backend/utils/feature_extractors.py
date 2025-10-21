import numpy as np
import librosa


def extract_mfcc(audio, sr, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1), np.std(mfcc, axis=1)


def extract_chroma(audio, sr, n_chroma=12):
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_chroma=n_chroma)
    return np.mean(chroma, axis=1)


def extract_spectral_centroid(audio, sr):
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    return np.mean(centroid), np.std(centroid)


def extract_spectral_rolloff(audio, sr):
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    return np.mean(rolloff)


def extract_zero_crossing_rate(audio):
    zcr = librosa.feature.zero_crossing_rate(audio)
    return np.mean(zcr)


def extract_tempo(audio, sr):
    try:
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        return float(tempo)
    except:
        return 0.0


def extract_rms(audio):
    rms = librosa.feature.rms(y=audio)
    return np.mean(rms)


def extract_spectral_bandwidth(audio, sr):
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    return np.mean(bandwidth)
