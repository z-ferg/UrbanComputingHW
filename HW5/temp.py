# Imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score

# Load the data and preprocess
data_path = f'{os.getcwd()}/city_bike_NYC/202401-citibike-tripdata/202401-citibike-tripdata_2.csv'
df = pd.read_csv(data_path)

t2_df = df.dropna(subset=['start_lat', 'start_lng', 'end_lat', 'end_lng'])
data = t2_df[['start_lat', 'start_lng', 'end_lat', 'end_lng']].values

# Sample data to speed it up
sample_df = df.sample(n=100000, random_state=42)
sample_df = sample_df.dropna(subset=['start_lat', 'start_lng', 'end_lat', 'end_lng'])
sample_data = sample_df[['start_lat', 'start_lng', 'end_lat', 'end_lng']].values

# !! ChatGPT Generated Code !!
# --- Step 1: Elbow + Silhouette Test using MiniBatchKMeans ---
inertias, sil_scores = [], []
K_range = range(2, 15)

for k in K_range:
    print(f"[Elbow] Running MiniBatchKMeans for k={k} ...")
    kmeans = MiniBatchKMeans(
        n_clusters=k,
        init='k-means++',
        batch_size=50000,
        n_init=5,
        max_iter=100,
        random_state=42
    )
    kmeans.fit(sample_data)
    inertias.append(kmeans.inertia_)

    if k < 10:
        idx = np.random.choice(sample_data.shape[0], 5000, replace=False)
        sil = silhouette_score(sample_data[idx], kmeans.predict(sample_data[idx]))
        sil_scores.append(sil)
    else:
        sil_scores.append(np.nan)

# --- Step 2: Plot Elbow & Silhouette ---
fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(K_range, inertias, 'o-', color='blue', label='Inertia (Elbow)')
ax1.set_xlabel('Number of clusters (k)')
ax1.set_ylabel('Inertia', color='blue')

ax2 = ax1.twinx()
ax2.plot(K_range, sil_scores, 's--', color='green', label='Silhouette')
ax2.set_ylabel('Silhouette Score', color='green')

plt.title('Elbow & Silhouette (MiniBatchKMeans on Sample)')
plt.show()

# Fit both KMeans models according to "elbow" in the plot
K = 6

vanilla_kmeans = KMeans(n_clusters=K, init='random', n_init=5, random_state=42)
vanilla_kmeans.fit(data)

special_kmeans = KMeans(n_clusters=K, init='k-means++', n_init=5, random_state=42)
special_kmeans.fit(data)

# !! ChatGPT Generated Code !!
# Sample for testing data
eval_sample_size = 50000
idx = np.random.choice(len(data), eval_sample_size, replace=False)
samples_subset = data[idx]

# Make predictions using the models
vanilla_labels = vanilla_kmeans.predict(samples_subset)
special_labels = special_kmeans.predict(samples_subset)

# Calculate and print the silhouette scores
vanilla_sil = silhouette_score(samples_subset, vanilla_labels)
special_sil = silhouette_score(samples_subset, special_labels)

print("\n=== Cluster Performance Summary ===")
print(f"Vanilla KMeans:  Inertia = {vanilla_kmeans.inertia_:.2f},  Silhouette = {vanilla_sil:.4f}")
print(f"K-Means++:       Inertia = {special_kmeans.inertia_:.2f},  Silhouette = {special_sil:.4f}")

# !! ChatGPT Generated Code !!

# Predict cluster labels
labels_vanilla = vanilla_kmeans.labels_
labels_special = special_kmeans.labels_

# Plot comparison side-by-side
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Vanilla K-Means
axes[0,0].scatter(t2_df['start_lng'], t2_df['start_lat'], c=labels_vanilla, s=10, cmap='tab10')
axes[0,0].set_title("Vanilla K-Means: Start Locations")
axes[0,0].set_xlabel("Longitude"); axes[0,0].set_ylabel("Latitude")

axes[0,1].scatter(t2_df['end_lng'], t2_df['end_lat'], c=labels_vanilla, s=10, cmap='tab10')
axes[0,1].set_title("Vanilla K-Means: End Locations")
axes[0,1].set_xlabel("Longitude"); axes[0,1].set_ylabel("Latitude")

# K-Means++
axes[1,0].scatter(t2_df['start_lng'], t2_df['start_lat'], c=labels_special, s=10, cmap='tab10')
axes[1,0].set_title("K-Means++: Start Locations")
axes[1,0].set_xlabel("Longitude"); axes[1,0].set_ylabel("Latitude")

axes[1,1].scatter(t2_df['end_lng'], t2_df['end_lat'], c=labels_special, s=10, cmap='tab10')
axes[1,1].set_title("K-Means++: End Locations")
axes[1,1].set_xlabel("Longitude"); axes[1,1].set_ylabel("Latitude")

plt.suptitle(f"NYC Citibike Clustering Comparison (K={K})")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(t2_df['start_lng'], t2_df['start_lat'], c=labels_special, s=10, cmap='tab10')
axes[0].set_title("Start Locations")
axes[0].set_xlabel("Longitude")
axes[0].set_ylabel("Latitude")

axes[1].scatter(t2_df['end_lng'], t2_df['end_lat'], c=labels_special, s=10, cmap='tab10')
axes[1].set_title("End Locations")
axes[1].set_xlabel("Longitude")
axes[1].set_ylabel("Latitude")

plt.suptitle("Special K-Means Clustering on NYC Citibike Trips")
plt.tight_layout()
plt.show()

print(f"Random Inertia: f{vanilla_kmeans.inertia_}")
print(f"K-means++ Inertia: f{special_kmeans.inertia_}")