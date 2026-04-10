import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
import numpy as np

# dataset load
digits = load_digits()
X = digits.data
y = digits.target

print("Dataset Loaded Successfully")
print("Shape:", X.shape)

# PCA to 2D
pca_2 = PCA(n_components=2)
X_pca_2 = pca_2.fit_transform(X)

# plot 2D projection
plt.scatter(X_pca_2[:, 0], X_pca_2[:, 1], c=y, cmap='viridis')

plt.title("PCA 2D Projection")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.savefig("pca_2d_plot.png")
plt.show()


# PCA to 50D
pca_50 = PCA(n_components=50)
X_pca_50 = pca_50.fit_transform(X)

print("Reduced Shape (50D):", X_pca_50.shape)


# cumulative explained variance
pca_full = PCA().fit(X)

cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

plt.plot(cumulative_variance)

plt.title("Cumulative Explained Variance")
plt.xlabel("Number of Components")
plt.ylabel("Variance Explained")

plt.savefig("variance_plot.png")
plt.show()


# components needed for 95% variance
components_95 = np.argmax(cumulative_variance >= 0.95) + 1

print("Components required for 95% variance:", components_95)