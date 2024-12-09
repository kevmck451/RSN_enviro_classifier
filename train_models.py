import numpy as np
import librosa
import os
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import json

import numpy as np
import librosa
from scipy.signal import welch, spectrogram

def extract_features(file_path, feature_type="mfcc", n_mfcc=13, n_fft=2048, hop_length=512, n_per_seg=256):
    y, sr = librosa.load(file_path, sr=None)

    if feature_type == "mfcc":
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfcc, axis=1)

    elif feature_type == "chroma":
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        return np.mean(chroma, axis=1)

    elif feature_type == "zcr":
        zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)
        return np.array([np.mean(zcr)])  # Ensure 1D

    elif feature_type == "spectral_contrast":
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        return np.mean(spectral_contrast, axis=1)

    elif feature_type == "rms":
        rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)
        return np.array([np.mean(rms)])  # Ensure 1D

    elif feature_type == "psd":
        f, psd = welch(y, fs=sr, nperseg=n_per_seg)
        return np.array([np.mean(psd)])  # Return summary of PSD

    elif feature_type == "spectral_plot":
        f, t, Sxx = spectrogram(y, fs=sr, nperseg=n_per_seg)
        spectral_amplitudes = np.mean(Sxx, axis=1)
        valid_idx = np.logical_and(f >= 10, f <= 200)
        spectral_values = spectral_amplitudes[valid_idx]
        return spectral_values

    elif feature_type == "combined":
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc), axis=1)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length), axis=1)
        zcr = np.array([np.mean(librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length))])
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length), axis=1)
        rms = np.array([np.mean(librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length))])
        f_psd, psd = welch(y, fs=sr, nperseg=n_per_seg)
        psd_summary = np.array([np.mean(psd)])
        f_spec, t, Sxx = spectrogram(y, fs=sr, nperseg=n_per_seg)
        spectral_amplitudes = np.mean(Sxx, axis=1)
        valid_idx = np.logical_and(f_spec >= 10, f_spec <= 200)
        spectral_values = spectral_amplitudes[valid_idx]

        return np.concatenate([mfcc, chroma, zcr, spectral_contrast, rms, psd_summary, spectral_values])
    else:
        raise ValueError(f"Unsupported feature type: {feature_type}")



# Prepare dataset
def prepare_data(data_dir, feature_type, **kwargs):
    features = []
    labels = []
    for label, env in enumerate(['car', 'bus', 'tram', 'train']):
        env_path = os.path.join(data_dir, env)
        for file_name in os.listdir(env_path):
            file_path = os.path.join(env_path, file_name)
            feature = extract_features(file_path, feature_type=feature_type, **kwargs)
            features.append(feature.flatten())  # Ensure features are 1D
            labels.append(label)
    return np.array(features), np.array(labels)


# Train Gaussian Mixture Models for each class
def train_gmm_models(X_train, y_train, num_classes, num_components=2):
    gmms = []
    for cls in range(num_classes):
        X_class = X_train[y_train == cls]
        if X_class.ndim == 1:
            X_class = X_class.reshape(-1, 1)  # Reshape to 2D if necessary
        gmm = GaussianMixture(n_components=num_components, covariance_type='full', random_state=42)
        gmm.fit(X_class)
        gmms.append(gmm)
    return gmms



def predict_classes_gmm(X_test, gmms):
    """
    Predicts the class labels for the test data using GMMs.

    Args:
        X_test (np.ndarray): Test data (samples x features).
        gmms (list): List of trained GMM models, one for each class.

    Returns:
        np.ndarray: Predicted class labels.
    """
    # Ensure X_test is 2D
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)

    # Compute log-likelihood for each class
    log_likelihoods = np.array([gmm.score_samples(X_test) for gmm in gmms]).T

    # Predict the class with the highest log-likelihood
    return np.argmax(log_likelihoods, axis=1)


# Train and evaluate models
def train_and_evaluate_models(data_dir, feature_types, output_gmm_file, output_gnb_file):
    """
    Train and evaluate GMM and GNB models on different feature types.

    Args:
        data_dir (str): Path to the dataset directory.
        feature_types (list): List of feature types to evaluate.
        output_gmm_file (str): File to save GMM results.
        output_gnb_file (str): File to save GNB results.
    """
    gmm_summary = {}
    gnb_summary = {}

    for feature_type in feature_types:
        # Prepare data
        X, y = prepare_data(data_dir, feature_type=feature_type)

        # Reshape if features are 1D
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

        # Train and evaluate GMMs
        num_classes = len(np.unique(y))
        gmms = train_gmm_models(X_train, y_train, num_classes)
        y_pred_gmm = predict_classes_gmm(X_test, gmms)
        report_gmm = classification_report(y_test, y_pred_gmm, target_names=['car', 'bus', 'tram', 'train'], output_dict=True)
        conf_matrix_gmm = confusion_matrix(y_test, y_pred_gmm)

        gmm_summary[feature_type] = {
            "classification_report": report_gmm,
            "confusion_matrix": conf_matrix_gmm.tolist(),
            "accuracy": report_gmm["accuracy"]
        }

        # Train and evaluate Naive Bayes
        clf = GaussianNB()
        clf.fit(X_train, y_train)
        y_pred_gnb = clf.predict(X_test)
        report_gnb = classification_report(y_test, y_pred_gnb, target_names=['car', 'bus', 'tram', 'train'], output_dict=True)
        conf_matrix_gnb = confusion_matrix(y_test, y_pred_gnb)

        gnb_summary[feature_type] = {
            "classification_report": report_gnb,
            "confusion_matrix": conf_matrix_gnb.tolist(),
            "accuracy": report_gnb["accuracy"]
        }

    # Save results to summary files
    with open(output_gmm_file, "w") as f:
        json.dump(gmm_summary, f, indent=4)
    with open(output_gnb_file, "w") as f:
        json.dump(gnb_summary, f, indent=4)


# Main
if __name__ == "__main__":
    data_dir = 'data'
    feature_types = ["mfcc", "chroma", "zcr", "spectral_contrast", "rms", "psd", "spectral_plot", "combined"]
    output_gmm_file = "GMM_summary.json"
    output_gnb_file = "GNB_summary.json"

    train_and_evaluate_models(data_dir, feature_types, output_gmm_file, output_gnb_file)
    print(f"Summaries saved to {output_gmm_file} and {output_gnb_file}")

