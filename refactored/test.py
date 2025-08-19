import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE


# load the memories for some objects and visualize them with t-SNE
def visualize(memories_folder):
    # load all npy files in folder along with their names
    # they are of form: <name>_memories.npy
    memories = []
    names = []
    for filename in os.listdir(memories_folder):
        if filename.endswith(".npy"):
            memories.append(np.load(os.path.join(memories_folder, filename)))
            base = os.path.splitext(filename)[0]
            for suffix in ("_memories", "_memory"):
                if base.endswith(suffix):
                    base = base[: -len(suffix)]
                    break
            names.append(base)
    memories = np.array(memories).squeeze() # Nx512
    print(f"Loaded {len(memories)} memories.")

    # each memory is a 500-dimensional vector,
    # use t-SNE to reduce the dimensionality and visualize them
    tsne = TSNE(n_components=2, perplexity=5)
    reduced = tsne.fit_transform(memories)

    plt.figure(figsize=(10, 10))
    ax = sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1])
    for i, label in enumerate(names):
        ax.text(reduced[i, 0], reduced[i, 1], label, fontsize=9, ha='center', va='center')
    plt.title("t-SNE visualization of memories")
    plt.tight_layout()
    plt.show()


# learn a partial object and see which memory it is closest to



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", type=str, required=True)
    args = parser.parse_args()

    visualize(args.visualize)
