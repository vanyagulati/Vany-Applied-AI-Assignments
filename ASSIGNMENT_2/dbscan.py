import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

# dataset load
data = pd.read_csv("network_data.csv")

print("Dataset Loaded Successfully")
print(data.head())

X = data[['packet_size', 'request_frequency']]

# DBSCAN apply
dbscan = DBSCAN(eps=500, min_samples=2)
labels = dbscan.fit_predict(X)

print("DBSCAN Labels:", labels)

# Plot DBSCAN result
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap='rainbow')
plt.title("DBSCAN Clustering")
plt.xlabel("Packet Size")
plt.ylabel("Request Frequency")

plt.savefig("dbscan_plot.png")
plt.show()


# KMeans comparison
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=kmeans_labels, cmap='viridis')
plt.title("KMeans Clustering Comparison")
plt.xlabel("Packet Size")
plt.ylabel("Request Frequency")

plt.savefig("kmeans_comparison_plot.png")
plt.show()