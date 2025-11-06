# Imports
import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

K=10

t1_df = pd.read_csv(f'{os.getcwd()}/city_bike_NYC/202401-citibike-tripdata/202401-citibike-tripdata_1.csv')
t2_df = pd.read_csv(f'{os.getcwd()}/city_bike_NYC/202401-citibike-tripdata/202401-citibike-tripdata_2.csv')

# From the piazza response it seemed like we only use the data_2 not the data_1
t2_df = t2_df.dropna(subset=['start_lat', 'start_lng', 'end_lat', 'end_lng'])

data = []
for _, x in t2_df.iterrows():
    data.append((x['start_lat'], x['start_lng'], x['end_lat'], x['end_lng']))

vanilla_kmeans = KMeans(n_clusters=K, init='random')
vanilla_kmeans.fit(data)

special_kmeans = KMeans(n_clusters=K, init='k-means++')
special_kmeans.fit(data)

import matplotlib.pyplot as plt

# Extract start locations
X_start = t2_df[['start_lat', 'start_lng']].values

# Predict cluster labels for start locations
labels_vanilla = vanilla_kmeans.predict(data)
labels_special = special_kmeans.predict(data)

# Plot vanilla k-means results
plt.figure(figsize=(8, 6))
plt.scatter(X_start[:, 1], X_start[:, 0], c=labels_vanilla, s=10, cmap='tab10')
plt.title("Vanilla K-Means Clustering (Start Locations)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()
