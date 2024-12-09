
from scipy.signal import welch
import numpy as np
import librosa



def extract_features(file_path, n_fft=2048, hop_length=512, n_mfcc=13, n_per_seg=256, freq_range=(10, 200)):
    """
    Extracts a combination of audio features from an audio file.

    Args:
        file_path (str): Path to the audio file.
        n_fft (int): Number of FFT components for spectral features.
        hop_length (int): Hop length for STFT and features.
        n_mfcc (int): Number of MFCC coefficients to compute.
        n_per_seg (int): Window length for Welch's method.
        freq_range (tuple): Frequency range for Welch's method (low, high).

    Returns:
        np.ndarray: Concatenated feature vector.
    """
    y, sr = librosa.load(file_path, sr=None)

    # 1. Power Spectral Density (Welch's Method)
    f, Pxx = welch(y, fs=sr, nperseg=n_per_seg)
    psd_indices = np.where((f >= freq_range[0]) & (f <= freq_range[1]))
    psd_feature = np.mean(Pxx[psd_indices])

    # 2. MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfcc_feature = np.mean(mfcc, axis=1)

    # 3. Chroma Features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    chroma_feature = np.mean(chroma, axis=1)

    # 4. Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)
    zcr_feature = np.mean(zcr)

    # 5. Spectral Contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    spectral_contrast_feature = np.mean(spectral_contrast, axis=1)

    # 6. RMS Energy
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)
    rms_feature = np.mean(rms)

    # Concatenate all features into a single feature vector
    feature_vector = np.concatenate([
        [psd_feature],  # PSD
        mfcc_feature,  # MFCC
        chroma_feature,  # Chroma
        [zcr_feature],  # Zero Crossing Rate
        spectral_contrast_feature,  # Spectral Contrast
        [rms_feature]  # RMS Energy
    ])

    return feature_vector
