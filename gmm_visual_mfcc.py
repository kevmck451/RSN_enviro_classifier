import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import librosa

# Feature extraction function
def extract_features(file_path, feature_type="mfcc", n_mfcc=13):
    y, sr = librosa.load(file_path, sr=None)
    if feature_type == "mfcc":
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfcc, axis=1)
    else:
        raise ValueError("Unsupported feature type")

def prepare_data(data_dir, feature_type="mfcc"):
    features = []
    labels = []
    for label, env in enumerate(['car', 'bus', 'tram', 'train']):
        env_path = os.path.join(data_dir, env)
        for file_name in os.listdir(env_path):
            file_path = os.path.join(env_path, file_name)
            try:
                feature = extract_features(file_path, feature_type=feature_type)
                features.append(feature)
                labels.append(label)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    # Ensure matching sizes
    features = np.array(features)
    labels = np.array(labels)
    if features.shape[0] != labels.shape[0]:
        raise ValueError(f"Mismatch in features ({features.shape[0]}) and labels ({labels.shape[0]})")
    return features, labels


def visualize_gmm_clusters(data_dir, feature_type="mfcc", num_components=4, zoom_multiplier=3):
    # Prepare data
    X, y = prepare_data(data_dir, feature_type=feature_type)

    # Class labels
    class_labels = ['car', 'bus', 'tram', 'train']

    # Reduce dimensionality to 2 for visualization if necessary
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X)
    else:
        X_reduced = X

    # Check size consistency
    if X_reduced.shape[0] != y.shape[0]:
        raise ValueError(f"Mismatch in data size: X_reduced has {X_reduced.shape[0]} samples, y has {y.shape[0]} labels.")

    # Fit GMM
    gmm = GaussianMixture(n_components=num_components, covariance_type='full', random_state=42)
    gmm.fit(X_reduced)

    # Extend grid range dynamically
    x_min, x_max = X_reduced[:, 0].min(), X_reduced[:, 0].max()
    y_min, y_max = X_reduced[:, 1].min(), X_reduced[:, 1].max()
    x_range = (x_max - x_min) * zoom_multiplier
    y_range = (y_max - y_min) * zoom_multiplier

    x = np.linspace(x_min - x_range, x_max + x_range, 300)
    y_grid = np.linspace(y_min - y_range, y_max + y_range, 300)
    X_grid, Y_grid = np.meshgrid(x, y_grid)
    XX = np.array([X_grid.ravel(), Y_grid.ravel()]).T
    Z = -gmm.score_samples(XX)  # Negative log probability
    Z = Z.reshape(X_grid.shape)

    # Plot data and GMM contours
    plt.figure(figsize=(12, 10))

    # Use the class labels for scatter plot
    for i, label in enumerate(class_labels):
        plt.scatter(
            X_reduced[y == i, 0],
            X_reduced[y == i, 1],
            label=label,
            s=15,
            alpha=0.6
        )

    contour = plt.contour(
        X_grid,
        Y_grid,
        Z,
        levels=np.logspace(0, 3, 15),  # Adjust levels for finer rings
        cmap='coolwarm',
        alpha=0.75
    )
    plt.clabel(contour, inline=1, fontsize=8, fmt="%.0f")  # Add labels to contour lines
    plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c='red', marker='x', s=100, label='GMM Centers')
    plt.title(f"GMM Clusters and Probability Density for {feature_type.capitalize()} Features")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()

# Example usage
data_dir = "data"  # Replace with your dataset directory
visualize_gmm_clusters(data_dir, feature_type="mfcc", num_components=4, zoom_multiplier=2)


