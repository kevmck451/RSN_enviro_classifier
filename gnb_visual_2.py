import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate synthetic 1D and 2D data
X, y = make_classification(
    n_samples=400,
    n_features=2,
    n_classes=2,
    n_clusters_per_class=1,
    n_informative=2,
    n_redundant=0,
    random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Plot 1D Feature Distribution
plt.figure(figsize=(12, 4))
for label in np.unique(y_train):
    plt.hist(
        X_train[y_train == label, 0],
        bins=15,
        alpha=0.5,
        label=f'Class {label}',
        density=True
    )
    mean, std = X_train[y_train == label, 0].mean(), X_train[y_train == label, 0].std()
    x_range = np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 100)
    plt.plot(x_range, 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x_range - mean) / std) ** 2), label=f'Class {label} Gaussian')

plt.title('1D Feature Distribution with Gaussian Fits')
plt.xlabel('Feature 1')
plt.ylabel('Density')
plt.legend()
plt.show()

# Decision Boundary in 2D Space
xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200),
    np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200)
)
Z = gnb.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.6, cmap='coolwarm')
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', edgecolor='k')
plt.title('Decision Boundary for Gaussian Naive Bayes')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
