import numpy as np
import librosa
import os
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Function to extract spectral contrast features
def extract_features(file_path, feature_type="spectral_contrast", n_fft=2048, hop_length=512):
    y, sr = librosa.load(file_path, sr=None)
    if feature_type == "spectral_contrast":
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        return np.mean(spectral_contrast, axis=1)  # Return mean spectral contrast
    else:
        raise ValueError(f"Unsupported feature type: {feature_type}")

# Function to prepare dataset
def prepare_data(data_dir, feature_type, **kwargs):
    features = []
    labels = []
    for label, env in enumerate(['car', 'bus', 'tram', 'train']):  # Class labels
        env_path = os.path.join(data_dir, env)
        for file_name in os.listdir(env_path):
            file_path = os.path.join(env_path, file_name)
            feature = extract_features(file_path, feature_type=feature_type, **kwargs)
            features.append(feature)
            labels.append(label)
    return np.array(features), np.array(labels)

# Main script
data_dir = "data"  # Path to your dataset folder
feature_type = "spectral_contrast"

# Load data
X, y = prepare_data(data_dir, feature_type=feature_type, n_fft=2048, hop_length=512)  # Extract spectral contrast
X = X[:, :2]  # Use the first two spectral contrast features for visualization
X = StandardScaler().fit_transform(X)  # Standardize features

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Generate grid for decision boundaries
xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200),
    np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200)
)
grid_points = np.c_[xx.ravel(), yy.ravel()]
probs = gnb.predict_proba(grid_points).reshape(xx.shape + (len(np.unique(y)),))

# Define class labels
class_labels = ['car', 'bus', 'tram', 'train']

# Plot 1D Feature Distributions
plt.figure(figsize=(12, 6))
for class_label in np.unique(y_train):
    subset = X_train[y_train == class_label, 0]
    mean, std = subset.mean(), subset.std()
    x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    gaussian = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mean) / std) ** 2)
    plt.plot(x_range, gaussian, label=f"{class_labels[class_label]} Gaussian")
    plt.hist(subset, bins=20, alpha=0.3, density=True, label=f"{class_labels[class_label]} Data")

plt.title("1D Feature Distributions with Gaussian Fits (Spectral Contrast 1)")
plt.xlabel("Spectral Contrast 1")
plt.ylabel("Density")
plt.legend()
plt.show()

# Plot 2D Decision Boundaries
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, np.argmax(probs, axis=2), alpha=0.4, cmap='coolwarm')
scatter = plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', edgecolor='k', alpha=0.7)
cbar = plt.colorbar(scatter)
cbar.set_ticks(range(len(class_labels)))
cbar.set_ticklabels(class_labels)
plt.title("GNB Decision Boundaries and Training Data (Spectral Contrast)")
plt.xlabel("Spectral Contrast 1")
plt.ylabel("Spectral Contrast 2")
plt.show()

# Plot Confidence Heatmaps
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for i, ax in enumerate(axes.ravel()):
    contour = ax.contourf(xx, yy, probs[..., i], alpha=0.7, cmap='coolwarm')
    scatter = ax.scatter(X_train[:, 0], X_train[:, 1], c=(y_train == i), cmap='viridis', edgecolor='k', alpha=0.7)
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label("Probability")
    ax.set_title(f"Confidence Heatmap for {class_labels[i]}")
    ax.set_xlabel("Spectral Contrast 1")
    ax.set_ylabel("Spectral Contrast 2")

plt.tight_layout()
plt.show()
