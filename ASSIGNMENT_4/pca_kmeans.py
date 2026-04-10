import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import adjusted_rand_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

print("Dataset Loaded Successfully")
print("Original Shape:", X.shape)


# KMeans BEFORE PCA
kmeans_before = KMeans(n_clusters=3, random_state=42)
labels_before = kmeans_before.fit_predict(X)

ari_before = adjusted_rand_score(y, labels_before)

print("ARI before PCA:", ari_before)


# Apply PCA (reduce to 2 components)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("Reduced Shape after PCA:", X_pca.shape)


# KMeans AFTER PCA
kmeans_after = KMeans(n_clusters=3, random_state=42)
labels_after = kmeans_after.fit_predict(X_pca)

ari_after = adjusted_rand_score(y, labels_after)

print("ARI after PCA:", ari_after)


# Plot BEFORE PCA clustering
plt.scatter(X[:, 0], X[:, 1], c=labels_before, cmap='viridis')

plt.title("KMeans Clustering BEFORE PCA")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.savefig("before_pca_plot.png")
plt.show()


# Plot AFTER PCA clustering
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_after, cmap='rainbow')

plt.title("KMeans Clustering AFTER PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.savefig("after_pca_plot.png")
plt.show()