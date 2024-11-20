import numpy as np
import librosa
import os
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load and extract frequency features using STFT
def extract_stft_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    stft = np.abs(librosa.stft(y, n_fft=2**12))  # Magnitude STFT
    stft_mean = np.mean(stft, axis=1)  # Average over time for each frequency bin
    return stft_mean

# Prepare data
def prepare_data(data_dir):
    features = []
    labels = []
    for label, env in enumerate(['car', 'bus', 'tram', 'train']):
        env_path = os.path.join(data_dir, env)
        for file_name in os.listdir(env_path):
            file_path = os.path.join(env_path, file_name)
            feature = extract_stft_features(file_path)
            features.append(feature)
            labels.append(label)
    return np.array(features), np.array(labels)

# Load data
data_dir = '../data'
X, y = prepare_data(data_dir)

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Reduce dimensionality with PCA
pca = PCA(n_components=10)  # Adjust components as needed
X_pca = pca.fit_transform(X)

# Perform spectral clustering
cluster_model = SpectralClustering(n_clusters=4, affinity='nearest_neighbors', random_state=42)
y_pred = cluster_model.fit_predict(X_pca)

# Map clusters to labels (manual mapping may be needed based on clustering results)
print(classification_report(y, y_pred, target_names=['car', 'bus', 'tram', 'train']))
