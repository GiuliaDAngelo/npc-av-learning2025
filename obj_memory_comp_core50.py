import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.special import softmax
import matplotlib
import os

matplotlib.use('Agg')  # For headless environments

# === Ensure output folder exists ===
output_dir = 'analysis/core50'
os.makedirs(output_dir, exist_ok=True)

# === Load memory representations from all sessions ===
data_arrays = []
valid_session_ids = []

for i in range(1, 7):
    path = f'/Users/giuliadangelo/Downloads/npc-av-learning/core50cropped/workingmemory/s{i}/'
    filenames = glob.glob(path + 'object*memory.npy')

    if not filenames:
        print(f"⚠️  No files found in Session {i}.")
        continue

    data = []
    for f in filenames:
        arr = np.load(f)
        if arr.size > 0:
            data.append(arr.squeeze())
        else:
            print(f"⚠️  Empty array in file: {f}")

    if data:
        stacked = np.array(data)
        if stacked.ndim == 3:
            stacked = stacked.squeeze()
        if stacked.ndim == 1:
            stacked = stacked[:, np.newaxis]
        data_arrays.append(stacked)
        valid_session_ids.append(i)
        print(f"✅ Session {i}: Loaded shape {stacked.shape}")
    else:
        print(f"❌ Skipping Session {i}: All arrays empty or invalid.")

# === Proceed only if valid sessions ===
n_sessions = len(data_arrays)
if n_sessions < 2:
    raise RuntimeError("Not enough valid sessions with data to run analysis.")

n_objects = data_arrays[0].shape[0]

# === 1. Pairwise Cosine Similarity Between Sessions ===
# Each plot compares all object representations between two sessions
# Bright diagonal = same objects match across sessions
for i in range(n_sessions):
    for j in range(i, n_sessions):
        sim_matrix = 1 - cdist(data_arrays[i], data_arrays[j], 'cosine')
        plt.figure(figsize=(6, 5))
        plt.imshow(sim_matrix, interpolation='none', cmap='plasma')
        plt.colorbar(label='Cosine Similarity')
        plt.xlabel(f'Objects in Session {valid_session_ids[j]}')
        plt.ylabel(f'Objects in Session {valid_session_ids[i]}')
        plt.title(f'Object Similarity Matrix\nSession {valid_session_ids[i]} vs {valid_session_ids[j]}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'sim_matrix_s{valid_session_ids[i]}_vs_s{valid_session_ids[j]}.png'))
        plt.close()

# === 2. Softmax Similarity: Average vs Each Session ===
# Measures how well each object in a session matches the average (template) object across sessions
# Each row is a distribution over which object best matches the average object
summary_data = np.mean(data_arrays, axis=0)
for i in range(n_sessions):
    sim_matrix = 1 - cdist(summary_data, data_arrays[i], 'cosine')
    sim_softmax = softmax(10 * sim_matrix, axis=1)
    plt.figure(figsize=(6, 5))
    plt.imshow(sim_softmax, interpolation='none', cmap='inferno')
    plt.colorbar(label='Softmax Cosine Similarity')
    plt.xlabel(f'Objects in Session {valid_session_ids[i]}')
    plt.ylabel('Average Template Object Index')
    plt.title(f'Softmax Similarity: Average vs Session {valid_session_ids[i]}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'softmax_avg_vs_s{valid_session_ids[i]}.png'))
    plt.close()


# === 3. Mean Same-Object Similarity Matrix ===
# Compares object k in session i with object k in session j
# High off-diagonal = stable object encoding across sessions
similarity_matrix = np.zeros((n_sessions, n_sessions))
for i in range(n_sessions):
    for j in range(n_sessions):
        sim = 1 - cdist(data_arrays[i], data_arrays[j], 'cosine')
        similarity_matrix[i, j] = np.mean(np.diag(sim))

plt.figure(figsize=(6, 5))
plt.imshow(similarity_matrix, interpolation='none', cmap='viridis')
plt.colorbar(label='Mean Cosine Similarity (Same Object)')
plt.xticks(range(n_sessions), [f'S{i}' for i in valid_session_ids])
plt.yticks(range(n_sessions), [f'S{i}' for i in valid_session_ids])
plt.title('Mean Same-Object Similarity Between Sessions')
plt.xlabel('Compared To')
plt.ylabel('Reference Session')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'mean_same_object_similarity_matrix.png'))
plt.close()

# === 4. Object Drift Over Sessions (vs Session 1) ===
# Tracks how each object changes across sessions compared to session 1
# Flat line near 1.0 = stable; downward drift = memory changes
object_drift = np.zeros((n_objects, n_sessions))
ref_session = data_arrays[0]
for obj_idx in range(n_objects):
    ref_vec = ref_session[obj_idx]
    for sess_idx in range(n_sessions):
        object_drift[obj_idx, sess_idx] = 1 - cdist([ref_vec], [data_arrays[sess_idx][obj_idx]], 'cosine')[0, 0]

plt.figure(figsize=(8, 5))
for obj_idx in range(n_objects):
    plt.plot(range(n_sessions), object_drift[obj_idx], alpha=0.5, label=f'Obj {obj_idx}' if obj_idx < 5 else None)
plt.xlabel('Session Index')
plt.ylabel('Cosine Similarity to Session 1')
plt.title('Object Representation Drift Over Sessions')
plt.grid(True)
plt.legend(loc='lower left', title='Example Objects', fontsize='small')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'object_drift_plot.png'))
plt.close()

# === 5. Intra-Session Similarity (Object Entanglement) ===
# Measures how similar objects are to each other within each session
# High value = entangled representation (less distinct objects)
intra_sim = []
for i in range(n_sessions):
    sim = 1 - cdist(data_arrays[i], data_arrays[i], 'cosine')
    mask = ~np.eye(n_objects, dtype=bool)
    avg_intra = np.mean(sim[mask])
    intra_sim.append(avg_intra)

plt.figure(figsize=(6, 4))
plt.plot(range(n_sessions), intra_sim, marker='o', linestyle='-', color='darkred')
plt.xticks(range(n_sessions), [f'S{i}' for i in valid_session_ids])
plt.xlabel('Session')
plt.ylabel('Mean Cosine Similarity (Object-Object)')
plt.title('Intra-Session Object Similarity')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'intra_session_similarity.png'))
plt.close()
