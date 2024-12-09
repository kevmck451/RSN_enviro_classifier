import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Generate synthetic dataset with 4 classes
X, y = make_classification(
    n_samples=500,
    n_features=2,
    n_classes=4,
    n_clusters_per_class=1,
    n_informative=2,
    n_redundant=0,
    random_state=42
)

# Standardize features for better visualization
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Generate grid for decision boundary
xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200),
    np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200)
)
grid_points = np.c_[xx.ravel(), yy.ravel()]
probs = gnb.predict_proba(grid_points).reshape(xx.shape + (4,))

# Plot 1D Feature Distributions
plt.figure(figsize=(12, 6))
for class_label in np.unique(y_train):
    subset = X_train[y_train == class_label, 0]
    mean, std = subset.mean(), subset.std()
    x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    gaussian = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mean) / std) ** 2)
    plt.plot(x_range, gaussian, label=f"Class {class_label} Gaussian")
    plt.hist(subset, bins=20, alpha=0.3, density=True, label=f"Class {class_label} Data")

plt.title("1D Feature Distributions with Gaussian Fits")
plt.xlabel("Feature 1")
plt.ylabel("Density")
plt.legend()
plt.show()

# Plot 2D Decision Boundaries
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, np.argmax(probs, axis=2), alpha=0.4, cmap='coolwarm')
scatter = plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', edgecolor='k', alpha=0.7)
plt.colorbar(scatter, label="Class")
plt.title("GNB Decision Boundaries and Training Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Plot Confidence Heatmaps for All Classes
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for i, ax in enumerate(axes.ravel()):
    contour = ax.contourf(xx, yy, probs[..., i], alpha=0.7, cmap='coolwarm')
    ax.scatter(X_train[:, 0], X_train[:, 1], c=(y_train == i), cmap='viridis', edgecolor='k', alpha=0.7)
    ax.set_title(f"Confidence Heatmap for Class {i}")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    plt.colorbar(contour, ax=ax)

plt.tight_layout()
plt.show()
