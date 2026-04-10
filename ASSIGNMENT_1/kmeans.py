import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# Dataset load karna
data = pd.read_csv("Mall_Customers.csv")

print("Dataset loaded successfully")
print(data.head())


# Important columns select karna
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]


# Elbow Method apply karna
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)


# Elbow graph banana
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')

plt.savefig("elbow_plot.png")
plt.show()


# KMeans clustering apply karna
kmeans = KMeans(n_clusters=5, random_state=42)
y_kmeans = kmeans.fit_predict(X)


# Silhouette Score calculate karna
score = silhouette_score(X, y_kmeans)

print("Silhouette Score:", score)


# Scatter plot banana
plt.scatter(X.iloc[:, 0], X.iloc[:, 1],
            c=y_kmeans, cmap='rainbow')

plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            s=200,
            c='black',
            label='Centroids')

plt.title("Customer Segmentation")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")

plt.legend()

plt.savefig("cluster_plot.png")
plt.show()