import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans

data = np.array([1,3,6,7]).reshape(-1, 1)
filler = np.array([0] * len(data))


k = 3
kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
kmeans.fit(data)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_
sse = kmeans.inertia_

print("Cluster labels:", labels)
print("Cluster centroids:", centroids)
print("SSE:", sse)