import numpy as np
import librosa
import os
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Function to extract MFCC features
def extract_features(file_path, feature_type="mfcc", n_mfcc=13):
    y, sr = librosa.load(file_path, sr=None)
    if feature_type == "mfcc":
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfcc, axis=1)  # Return mean MFCCs
    else:
        raise ValueError(f"Unsupported feature type: {feature_type}")

# Function to prepare dataset
def prepare_data(data_dir, feature_type, n_mfcc=13):
    features = []
    labels = []
    for label, env in enumerate(['car', 'bus', 'tram', 'train']):  # Class labels
        env_path = os.path.join(data_dir, env)
        for file_name in os.listdir(env_path):
            file_path = os.path.join(env_path, file_name)
            feature = extract_features(file_path, feature_type=feature_type, n_mfcc=n_mfcc)
            features.append(feature)
            labels.append(label)
    return np.array(features), np.array(labels)

# Main script
data_dir = "data"  # Path to your dataset folder
feature_type = "mfcc"

# Load data
X, y = prepare_data(data_dir, feature_type=feature_type, n_mfcc=2)  # Extract 2 MFCC coefficients
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

# Plot 1D Feature Distributions
plt.figure(figsize=(12, 6))
for class_label in np.unique(y_train):
    subset = X_train[y_train == class_label, 0]
    mean, std = subset.mean(), subset.std()
    x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    gaussian = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mean) / std) ** 2)
    plt.plot(x_range, gaussian, label=f"Class {class_label} Gaussian")
    plt.hist(subset, bins=20, alpha=0.3, density=True, label=f"Class {class_label} Data")

plt.title("1D Feature Distributions with Gaussian Fits (MFCC 1)")
plt.xlabel("MFCC 1")
plt.ylabel("Density")
plt.legend()
plt.show()

# Plot 2D Decision Boundaries
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, np.argmax(probs, axis=2), alpha=0.4, cmap='coolwarm')
scatter = plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', edgecolor='k', alpha=0.7)
plt.colorbar(scatter, label="Class")
plt.title("GNB Decision Boundaries and Training Data (MFCC)")
plt.xlabel("MFCC 1")
plt.ylabel("MFCC 2")
plt.show()

# Plot Confidence Heatmaps
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for i, ax in enumerate(axes.ravel()):
    contour = ax.contourf(xx, yy, probs[..., i], alpha=0.7, cmap='coolwarm')
    ax.scatter(X_train[:, 0], X_train[:, 1], c=(y_train == i), cmap='viridis', edgecolor='k', alpha=0.7)
    ax.set_title(f"Confidence Heatmap for Class {i}")
    ax.set_xlabel("MFCC 1")
    ax.set_ylabel("MFCC 2")
    plt.colorbar(contour, ax=ax)

plt.tight_layout()
plt.show()
