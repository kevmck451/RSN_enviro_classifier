from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

import numpy as np

from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

# Generate synthetic audio feature data for four environments
n_samples = 400
centers = [[0, 0], [4, 4], [-4, -4], [0, 5]]
cluster_std = [0.8, 1.0, 0.9, 0.7]

X, y_true = make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std, random_state=42)

# Fit Gaussian Mixture Model
gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=42)
gmm.fit(X)

# Predict cluster labels
y_gmm = gmm.predict(X)


# Create a grid for probability density visualization
x = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
y = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100)
X_grid, Y_grid = np.meshgrid(x, y)
XX = np.array([X_grid.ravel(), Y_grid.ravel()]).T

# Compute probability density for each point in the grid
Z = -gmm.score_samples(XX)  # Negative log probability
Z = Z.reshape(X_grid.shape)

# Plot data with probability density contours
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', s=15, alpha=0.6, label="Data Points")
ax.contour(X_grid, Y_grid, Z, levels=np.logspace(0, 2, 10), norm=LogNorm(), cmap='coolwarm', alpha=0.75)

# Overlay GMM means
ax.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c='red', marker='x', s=100, label='GMM Centers')
ax.set_title("GMM with Probability Contours", fontsize=14)
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.legend()
plt.show()
