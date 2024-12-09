import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

# Generate synthetic data for classification
X, y = make_classification(
    n_samples=400,
    n_features=2,
    n_classes=4,
    n_clusters_per_class=1,
    n_informative=2,  # Ensure this is <= n_features
    n_redundant=0,    # No redundant features
    random_state=42
)

# Standardize the features for better visualization
X = StandardScaler().fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Create a grid for decision boundary visualization
xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200),
    np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200)
)

# Predict probabilities on the grid
Z = gnb.predict_proba(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape + (Z.shape[1],))

# Plot decision boundaries and confidence heatmaps
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Scatter plot of the features
scatter = axs[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', edgecolor='k', alpha=0.7)
axs[0].set_title('Scatter Plot of Features')
axs[0].set_xlabel('Feature 1')
axs[0].set_ylabel('Feature 2')
axs[0].legend(*scatter.legend_elements(), title="Classes")

# Heatmap of class confidence for each class
for i, label in enumerate(["Class 0", "Class 1", "Class 2", "Class 3"]):
    contour = axs[1].contourf(xx, yy, Z[..., i], alpha=0.5, levels=10, cmap='coolwarm')
    axs[1].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', edgecolor='k', alpha=0.7)
    axs[1].set_title(f'Decision Boundaries & Confidence for {label}')
    axs[1].set_xlabel('Feature 1')
    axs[1].set_ylabel('Feature 2')

plt.tight_layout()
plt.show()
