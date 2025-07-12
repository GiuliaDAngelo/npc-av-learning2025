import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, cosine
import matplotlib
from transformers import CLIPProcessor, CLIPModel


matplotlib.use('Agg')

# === Optional ML tools ===
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# === Setup ===
data_dir = '/Users/giuliadangelo/Downloads/npc-av-learning/CRIB/workingmemory/'
output_dir = 'analysis/CRIB'
os.makedirs(output_dir, exist_ok=True)

# === === ======
# === Memory ===
# === === ======


clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

# === Load all object memory vectors ===
filenames = glob.glob(os.path.join(data_dir, '*memory.npy'))

if not filenames:
    raise FileNotFoundError("❌ No memory files found!")

object_vectors = []
for f in filenames:
    arr = np.load(f)
    if arr.size > 0:
        arr = arr.squeeze()
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        object_vectors.append(arr)
    else:
        print(f"⚠️ Empty array in file: {f}")

data = np.vstack(object_vectors)
n_objects, feature_dim = data.shape
print(f"✅ Loaded {n_objects} objects with vector size {feature_dim}")

# === Cosine Similarity Matrix Between All Pairs ===
cosine_sim_matrix = 1 - cdist(data, data, metric='cosine')

plt.figure(figsize=(8, 7))
plt.imshow(cosine_sim_matrix, interpolation='none', cmap='viridis')
plt.colorbar(label='Cosine Similarity')
plt.title('Pairwise Cosine Similarity Between Objects')
plt.xlabel('Object Index')
plt.ylabel('Object Index')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'cosine_similarity_matrix.png'))
plt.close()

# === Similarity to Average Vector ===
average_vector = np.mean(data, axis=0)
cosine_to_avg = np.array([1 - cosine(obj, average_vector) for obj in data])

plt.figure(figsize=(8, 5))
plt.plot(range(n_objects), cosine_to_avg, marker='o', linestyle='-', color='navy')
plt.xlabel('Object Index')
plt.ylabel('Cosine Similarity to Average')
plt.title('Similarity of Each Object to Average Memory Vector')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'cosine_to_average.png'))
plt.close()

# === Histogram: Distribution of Similarities to Average ===
plt.figure(figsize=(6, 4))
plt.hist(cosine_to_avg, bins=20, color='purple', edgecolor='black')
plt.xlabel('Cosine Similarity to Average')
plt.ylabel('Count')
plt.title('Distribution of Cosine Similarity to Average')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'hist_cosine_to_average.png'))
plt.close()

# === PCA: Visualize Memory Space in 2D ===
pca = PCA(n_components=2)
pca_result = pca.fit_transform(data)

plt.figure(figsize=(6, 5))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=cosine_to_avg, cmap='viridis', edgecolor='k')
plt.colorbar(label='Cosine Similarity to Avg')
plt.title('PCA of Memory Vectors (colored by similarity to avg)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'pca_cosine_colored.png'))
plt.close()

# === Elbow Method: Determine optimal number of clusters ===
inertias = []
k_range = range(1, 15)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(data)
    inertias.append(km.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(k_range, inertias, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'elbow_plot.png'))
plt.close()

# === KMeans Clustering (e.g., with k=3 for now) ===
n_clusters = 14
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(data)

plt.figure(figsize=(6, 5))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap='tab10', edgecolor='k')
plt.title(f'KMeans Clustering (k={n_clusters}) on Memory Vectors')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'pca_kmeans_clusters.png'))
plt.close()

# === Sorted Similarity Matrix by Similarity to Average ===
sorted_indices = np.argsort(-cosine_to_avg)
sorted_sim_matrix = cosine_sim_matrix[sorted_indices][:, sorted_indices]

plt.figure(figsize=(8, 7))
plt.imshow(sorted_sim_matrix, interpolation='none', cmap='viridis')
plt.colorbar(label='Cosine Similarity')
plt.title('Pairwise Cosine Similarity (Sorted by Similarity to Avg)')
plt.xlabel('Sorted Object Index')
plt.ylabel('Sorted Object Index')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'cosine_matrix_sorted.png'))
plt.close()

print("✅ All analyses completed and visualizations saved.")


