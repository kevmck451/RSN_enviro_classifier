import numpy as np
import librosa
import os
from sklearn.mixture import GaussianMixture
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import json

# Define features to extract
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
        return np.mean(zcr)
    elif feature_type == "spectral_contrast":
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        return np.mean(spectral_contrast, axis=1)
    elif feature_type == "rms":
        rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)
        return np.mean(rms)
    elif feature_type == "combined":
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)
        return np.concatenate([np.mean(mfcc, axis=1), np.mean(chroma, axis=1),
                               [np.mean(zcr)], np.mean(spectral_contrast, axis=1),
                               [np.mean(rms)]])
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
            features.append(feature)
            labels.append(label)
    return np.array(features), np.array(labels)

# Train Gaussian Mixture Models for each class
def train_gmm_models(X_train, y_train, num_classes, num_components=2):
    gmms = []
    for cls in range(num_classes):
        X_class = X_train[y_train == cls]
        gmm = GaussianMixture(n_components=num_components, covariance_type='full', random_state=42)
        gmm.fit(X_class)
        gmms.append(gmm)
    return gmms

# Predict class labels
def predict_classes(X_test, gmms):
    log_likelihoods = np.array([gmm.score_samples(X_test) for gmm in gmms]).T
    return np.argmax(log_likelihoods, axis=1)

# Save results to a summary file
def save_summary(summary, output_path):
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=4)

# Main
if __name__ == "__main__":
    data_dir = 'data'
    feature_types = ["mfcc", "chroma", "zcr", "spectral_contrast", "rms", "combined"]
    output_file = "GMM_summary.json"

    summary = {}

    for feature_type in feature_types:
        # Prepare data
        X, y = prepare_data(data_dir, feature_type=feature_type)

        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

        # Train GMMs
        num_classes = len(np.unique(y))
        gmms = train_gmm_models(X_train, y_train, num_classes)

        # Predict and evaluate
        y_pred = predict_classes(X_test, gmms)
        report = classification_report(y_test, y_pred, target_names=['car', 'bus', 'tram', 'train'], output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Save results
        summary[feature_type] = {
            "classification_report": report,
            "confusion_matrix": conf_matrix.tolist(),
            "accuracy": report["accuracy"]
        }

    save_summary(summary, output_file)
    print(f"Summary saved to {output_file}")
