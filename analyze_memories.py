#!/usr/bin/env python3
"""
Analyze object memories using dimensionality reduction (t-SNE, PCA, UMAP).
Loads memories from JSON files and creates visualizations.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import argparse
import os


def load_memories(json_path):
    """Load memories from a JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Handle both old format (just memories) and new format (with classes)
    if 'memories' in data:
        memories_dict = data['memories']
        classes_dict = data.get('classes', {})
    else:
        # Old format - just memories
        memories_dict = data
        classes_dict = {}

    # Convert lists back to numpy arrays
    for obj_name in memories_dict:
        memories_dict[obj_name] = np.array(memories_dict[obj_name])

    return memories_dict, classes_dict


def analyze_memories(memories_dict, method='tsne', output_dir='analysis', classes_dict=None):
    """
    Perform dimensionality reduction on memories and create visualizations.

    Args:
        memories_dict: Dictionary mapping object names to memory vectors
        method: 'tsne', 'pca', or 'both'
        output_dir: Directory to save plots
        classes_dict: Optional dictionary mapping object names to class labels
    """
    os.makedirs(output_dir, exist_ok=True)

    # Prepare data
    object_names = list(memories_dict.keys())

    # Sort objects by class for better visualization
    if classes_dict:
        # Sort by class, then by instance name
        object_names_sorted = sorted(object_names, key=lambda name: (classes_dict.get(name, 'unknown'), name))
        object_classes = [classes_dict.get(name, 'unknown') for name in object_names_sorted]
        unique_classes = sorted(set(object_classes))
        class_to_color = {cls: plt.cm.tab10(i % 10) for i, cls in enumerate(unique_classes)}
        colors = [class_to_color[cls] for cls in object_classes]
    else:
        object_names_sorted = object_names
        colors = 'blue'
        object_classes = None

    # Reorder memory vectors according to sorted names
    memory_vectors = np.array([memories_dict[name] for name in object_names_sorted])

    print(f"Analyzing {len(object_names_sorted)} objects")
    print(f"Memory vector dimension: {memory_vectors.shape[1]}")

    # Compute pairwise similarities
    print("\nComputing pairwise cosine similarities...")
    similarities = np.zeros((len(object_names_sorted), len(object_names_sorted)))
    for i in range(len(object_names_sorted)):
        for j in range(len(object_names_sorted)):
            # Cosine similarity
            dot_product = np.dot(memory_vectors[i], memory_vectors[j])
            norm_i = np.linalg.norm(memory_vectors[i])
            norm_j = np.linalg.norm(memory_vectors[j])
            similarities[i, j] = dot_product / (norm_i * norm_j) if norm_i > 0 and norm_j > 0 else 0

    # Print similarity matrix (sorted by class)
    print("\nCosine Similarity Matrix (sorted by class):")
    print("".join([f"{name:12s}" for name in object_names_sorted]))
    for i, name in enumerate(object_names_sorted):
        print(f"{name:12s}" + "".join([f"{similarities[i, j]:12.4f}" for j in range(len(object_names_sorted))]))

    # Create similarity heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(similarities, cmap='viridis', aspect='auto')
    plt.colorbar(im, label='Cosine Similarity', ax=ax)
    ax.set_xticks(range(len(object_names_sorted)))
    ax.set_xticklabels(object_names_sorted, rotation=45, ha='right')
    ax.set_yticks(range(len(object_names_sorted)))
    ax.set_yticklabels(object_names_sorted)
    ax.set_title('Pairwise Cosine Similarity of Object Memories (Sorted by Class)')

    # Add grid lines between classes if class info is available
    if classes_dict and len(unique_classes) > 1:
        class_boundaries = []
        current_class = object_classes[0]
        for i, cls in enumerate(object_classes[1:], 1):
            if cls != current_class:
                class_boundaries.append(i - 0.5)
                current_class = cls

        # Draw lines between classes
        for boundary in class_boundaries:
            ax.axhline(y=boundary, color='white', linewidth=2, alpha=0.7)
            ax.axvline(x=boundary, color='white', linewidth=2, alpha=0.7)

    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, 'similarity_heatmap.png')
    plt.savefig(heatmap_path, dpi=150)
    print(f"\n✓ Similarity heatmap saved to {heatmap_path}")
    plt.close()

    # Perform dimensionality reduction
    if method in ['pca', 'both']:
        print("\nPerforming PCA...")
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(memory_vectors)

        # Plot PCA
        plt.figure(figsize=(12, 8))
        if classes_dict:
            for cls in unique_classes:
                mask = [object_classes[i] == cls for i in range(len(object_names_sorted))]
                plt.scatter(pca_result[mask, 0], pca_result[mask, 1],
                           c=[class_to_color[cls]], s=100, alpha=0.6, label=cls)
        else:
            plt.scatter(pca_result[:, 0], pca_result[:, 1], s=100, alpha=0.6, c=colors)

        for i, name in enumerate(object_names_sorted):
            plt.annotate(name, (pca_result[i, 0], pca_result[i, 1]),
                        fontsize=10, ha='center', va='bottom')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title('PCA Projection of Object Memories')
        if classes_dict:
            plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        pca_path = os.path.join(output_dir, 'pca_projection.png')
        plt.savefig(pca_path, dpi=150)
        print(f"✓ PCA plot saved to {pca_path}")
        plt.close()

    if method in ['tsne', 'both']:
        print("\nPerforming t-SNE (this may take a moment)...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(object_names_sorted) - 1))
        tsne_result = tsne.fit_transform(memory_vectors)

        # Plot t-SNE
        plt.figure(figsize=(12, 8))
        if classes_dict:
            for cls in unique_classes:
                mask = [object_classes[i] == cls for i in range(len(object_names_sorted))]
                plt.scatter(tsne_result[mask, 0], tsne_result[mask, 1],
                           c=[class_to_color[cls]], s=100, alpha=0.6, label=cls)
        else:
            plt.scatter(tsne_result[:, 0], tsne_result[:, 1], s=100, alpha=0.6, c=colors)

        for i, name in enumerate(object_names_sorted):
            plt.annotate(name, (tsne_result[i, 0], tsne_result[i, 1]),
                        fontsize=10, ha='center', va='bottom')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.title('t-SNE Projection of Object Memories')
        if classes_dict:
            plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        tsne_path = os.path.join(output_dir, 'tsne_projection.png')
        plt.savefig(tsne_path, dpi=150)
        print(f"✓ t-SNE plot saved to {tsne_path}")
        plt.close()

    # Try UMAP if available
    try:
        import umap
        print("\nPerforming UMAP...")
        umap_model = umap.UMAP(n_components=2, random_state=42,
                               n_neighbors=min(15, len(object_names_sorted) - 1))
        umap_result = umap_model.fit_transform(memory_vectors)

        # Plot UMAP
        plt.figure(figsize=(12, 8))
        if classes_dict:
            for cls in unique_classes:
                mask = [object_classes[i] == cls for i in range(len(object_names_sorted))]
                plt.scatter(umap_result[mask, 0], umap_result[mask, 1],
                           c=[class_to_color[cls]], s=100, alpha=0.6, label=cls)
        else:
            plt.scatter(umap_result[:, 0], umap_result[:, 1], s=100, alpha=0.6, c=colors)

        for i, name in enumerate(object_names_sorted):
            plt.annotate(name, (umap_result[i, 0], umap_result[i, 1]),
                        fontsize=10, ha='center', va='bottom')
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        plt.title('UMAP Projection of Object Memories')
        if classes_dict:
            plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        umap_path = os.path.join(output_dir, 'umap_projection.png')
        plt.savefig(umap_path, dpi=150)
        print(f"✓ UMAP plot saved to {umap_path}")
        plt.close()
    except ImportError:
        print("\nNote: UMAP not available. Install with: pip install umap-learn")

    print("\n" + "="*80)
    print("✓ Analysis complete!")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Analyze object memories with dimensionality reduction.')
    parser.add_argument('memory_file', type=str, help='Path to memories JSON file.')
    parser.add_argument('--method', type=str, default='both', choices=['tsne', 'pca', 'both'],
                       help='Dimensionality reduction method (default: both).')
    parser.add_argument('--output_dir', type=str, default='analysis',
                       help='Directory to save analysis plots (default: analysis).')
    args = parser.parse_args()

    if not os.path.exists(args.memory_file):
        print(f"Error: Memory file '{args.memory_file}' not found.")
        return

    print("="*80)
    print(f"Loading memories from {args.memory_file}...")
    print("="*80)

    memories, classes = load_memories(args.memory_file)

    if classes:
        unique_classes = sorted(set(classes.values()))
        print(f"\nDetected {len(unique_classes)} unique classes:")
        for cls in unique_classes:
            instances = [name for name, c in classes.items() if c == cls]
            print(f"  {cls}: {len(instances)} instance(s) - {instances}")
        print()

    analyze_memories(memories, method=args.method, output_dir=args.output_dir, classes_dict=classes)


if __name__ == "__main__":
    main()
