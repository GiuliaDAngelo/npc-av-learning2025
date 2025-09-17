"""
Working Memory Visualization and Analysis
Loads saved working memory representations and visualizes them in reduced dimensional space using t-SNE
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import json
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
from collections import defaultdict

class WorkingMemoryAnalyzer:
    """Analyzes and visualizes working memory representations"""
    
    def __init__(self, memory_dir, max_objects=10):
        self.memory_dir = memory_dir
        self.max_objects = max_objects
        self.memories = {}
        self.features = {}
        self.object_names = []
        
    def load_working_memories(self):
        """Load working memory files from the specified directory"""
        if not os.path.exists(self.memory_dir):
            print(f"❌ Memory directory not found: {self.memory_dir}")
            return False
            
        # Find memory files
        memory_files = [f for f in os.listdir(self.memory_dir) if f.endswith('_memory.npy')]
        feature_files = [f for f in os.listdir(self.memory_dir) if f.endswith('_image_features.npy')]
        
        if not memory_files:
            print(f"❌ No memory files found in {self.memory_dir}")
            return False
            
        print(f"Found {len(memory_files)} memory files")
        
        # Load up to max_objects
        loaded_count = 0
        for memory_file in sorted(memory_files)[:self.max_objects]:
            obj_name = memory_file.replace('_memory.npy', '')
            memory_path = os.path.join(self.memory_dir, memory_file)
            feature_path = os.path.join(self.memory_dir, f'{obj_name}_image_features.npy')
            
            try:
                # Load working memory
                memory = np.load(memory_path)
                self.memories[obj_name] = memory
                
                # Load image features if available
                if os.path.exists(feature_path):
                    features = np.load(feature_path)
                    self.features[obj_name] = features
                
                self.object_names.append(obj_name)
                print(f"  ✅ Loaded {obj_name}: memory shape {memory.shape}")
                loaded_count += 1
                
            except Exception as e:
                print(f"  ❌ Error loading {obj_name}: {e}")
                
        print(f"Successfully loaded {loaded_count} working memories")
        return loaded_count > 0
    
    def prepare_data_for_visualization(self):
        """Prepare working memory data for dimensionality reduction"""
        if not self.memories:
            print("❌ No memories loaded")
            return None, None
            
        # Stack all working memories
        memory_data = []
        labels = []
        
        for obj_name in self.object_names:
            if obj_name in self.memories:
                memory = self.memories[obj_name]
                
                # Handle complex SSP vectors - take real part if complex
                if np.iscomplexobj(memory):
                    memory = np.real(memory)
                
                # Flatten if multi-dimensional
                if memory.ndim > 1:
                    memory = memory.flatten()
                
                memory_data.append(memory)
                labels.append(obj_name)
        
        if not memory_data:
            print("❌ No valid memory data to visualize")
            return None, None
            
        memory_matrix = np.vstack(memory_data)
        print(f"Memory matrix shape: {memory_matrix.shape}")
        
        return memory_matrix, labels
    
    def visualize_with_tsne(self, perplexity=5, random_state=42):
        """Create t-SNE visualization of working memories"""
        memory_matrix, labels = self.prepare_data_for_visualization()
        
        if memory_matrix is None:
            return
        
        print("Running t-SNE dimensionality reduction...")
        
        # Adjust perplexity based on number of samples
        n_samples = len(labels)
        perplexity = min(perplexity, max(1, n_samples - 1))
        
        try:
            # Apply PCA first to reduce dimensionality if needed
            if memory_matrix.shape[1] > 50:
                print("Applying PCA preprocessing...")
                pca = PCA(n_components=50, random_state=random_state)
                memory_matrix = pca.fit_transform(memory_matrix)
                print(f"PCA reduced to {memory_matrix.shape[1]} dimensions")
            
            # Apply t-SNE
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, 
                       n_iter=1000, verbose=1)
            embeddings_2d = tsne.fit_transform(memory_matrix)
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            
            # Generate colors for each object
            colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
            
            # Plot each object
            for i, (label, color) in enumerate(zip(labels, colors)):
                plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], 
                           c=[color], s=100, alpha=0.7, label=label)
                
                # Add text labels
                plt.annotate(label, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                           xytext=(5, 5), textcoords='offset points', 
                           fontsize=9, alpha=0.8)
            
            plt.title(f'Working Memory Representations (t-SNE)\n{len(labels)} Objects')
            plt.xlabel('t-SNE Dimension 1')
            plt.ylabel('t-SNE Dimension 2')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save plot
            output_path = 'working_memory_tsne_visualization.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✅ Visualization saved to {output_path}")
            plt.show()
            
            return embeddings_2d, labels
            
        except Exception as e:
            print(f"❌ Error during t-SNE visualization: {e}")
            return None, None
    
    def analyze_memory_properties(self):
        """Analyze properties of the working memories"""
        if not self.memories:
            print("❌ No memories to analyze")
            return
            
        print("\n" + "="*60)
        print("WORKING MEMORY ANALYSIS")
        print("="*60)
        
        # Analyze memory dimensions and properties
        memory_stats = {}
        
        for obj_name in self.object_names:
            if obj_name in self.memories:
                memory = self.memories[obj_name]
                
                # Handle complex data
                if np.iscomplexobj(memory):
                    real_part = np.real(memory)
                    imag_part = np.imag(memory)
                    magnitude = np.abs(memory)
                else:
                    real_part = memory
                    imag_part = None
                    magnitude = np.abs(memory)
                
                stats = {
                    'shape': memory.shape,
                    'is_complex': np.iscomplexobj(memory),
                    'mean_magnitude': np.mean(magnitude),
                    'std_magnitude': np.std(magnitude),
                    'max_magnitude': np.max(magnitude),
                    'sparsity': np.sum(magnitude < 1e-6) / magnitude.size
                }
                
                if imag_part is not None:
                    stats['mean_real'] = np.mean(real_part)
                    stats['mean_imag'] = np.mean(imag_part)
                
                memory_stats[obj_name] = stats
        
        # Print analysis
        for obj_name, stats in memory_stats.items():
            print(f"\n{obj_name}:")
            print(f"  Shape: {stats['shape']}")
            print(f"  Complex: {stats['is_complex']}")
            print(f"  Mean magnitude: {stats['mean_magnitude']:.4f}")
            print(f"  Std magnitude: {stats['std_magnitude']:.4f}")
            print(f"  Max magnitude: {stats['max_magnitude']:.4f}")
            print(f"  Sparsity: {stats['sparsity']:.2%}")
            
            if 'mean_real' in stats:
                print(f"  Mean real: {stats['mean_real']:.4f}")
                print(f"  Mean imag: {stats['mean_imag']:.4f}")
        
        # Save analysis
        with open('working_memory_analysis.json', 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            json_stats = {}
            for obj_name, stats in memory_stats.items():
                json_stats[obj_name] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else 
                       float(v) if isinstance(v, np.floating) else 
                       int(v) if isinstance(v, np.integer) else v
                    for k, v in stats.items()
                }
            json.dump(json_stats, f, indent=2)
        
        print(f"\n✅ Analysis saved to 'working_memory_analysis.json'")
        
        return memory_stats
    
    def compare_with_image_features(self):
        """Compare working memories with direct image features if available"""
        if not self.features:
            print("No image features available for comparison")
            return
            
        print("\n" + "="*60)
        print("WORKING MEMORY vs IMAGE FEATURES COMPARISON")
        print("="*60)
        
        # Prepare data for both types
        memory_data = []
        feature_data = []
        common_objects = []
        
        for obj_name in self.object_names:
            if obj_name in self.memories and obj_name in self.features:
                memory = self.memories[obj_name]
                features = self.features[obj_name]
                
                # Process memory
                if np.iscomplexobj(memory):
                    memory = np.real(memory)
                if memory.ndim > 1:
                    memory = memory.flatten()
                
                # Process features
                if features.ndim > 1:
                    features = features.flatten()
                
                memory_data.append(memory)
                feature_data.append(features)
                common_objects.append(obj_name)
        
        if len(common_objects) < 2:
            print("Not enough objects with both memory and features for comparison")
            return
        
        # Create comparison visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # t-SNE for working memories
        if len(memory_data) > 1:
            memory_matrix = np.vstack(memory_data)
            if memory_matrix.shape[1] > 50:
                pca_mem = PCA(n_components=50)
                memory_matrix = pca_mem.fit_transform(memory_matrix)
            
            tsne_mem = TSNE(n_components=2, perplexity=min(3, len(common_objects)-1))
            mem_2d = tsne_mem.fit_transform(memory_matrix)
            
            colors = plt.cm.tab10(np.linspace(0, 1, len(common_objects)))
            for i, (obj, color) in enumerate(zip(common_objects, colors)):
                ax1.scatter(mem_2d[i, 0], mem_2d[i, 1], c=[color], s=100, alpha=0.7)
                ax1.annotate(obj, (mem_2d[i, 0], mem_2d[i, 1]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
            
            ax1.set_title('Working Memory t-SNE')
            ax1.grid(True, alpha=0.3)
        
        # t-SNE for image features
        if len(feature_data) > 1:
            feature_matrix = np.vstack(feature_data)
            if feature_matrix.shape[1] > 50:
                pca_feat = PCA(n_components=50)
                feature_matrix = pca_feat.fit_transform(feature_matrix)
            
            tsne_feat = TSNE(n_components=2, perplexity=min(3, len(common_objects)-1))
            feat_2d = tsne_feat.fit_transform(feature_matrix)
            
            for i, (obj, color) in enumerate(zip(common_objects, colors)):
                ax2.scatter(feat_2d[i, 0], feat_2d[i, 1], c=[color], s=100, alpha=0.7)
                ax2.annotate(obj, (feat_2d[i, 0], feat_2d[i, 1]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
            
            ax2.set_title('Image Features t-SNE')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('memory_vs_features_comparison.png', dpi=150, bbox_inches='tight')
        print("✅ Comparison visualization saved to 'memory_vs_features_comparison.png'")
        plt.show()


def main():
    """Main analysis pipeline"""
    
    # Configuration - update these paths as needed
    MEMORY_DIR = "/Users/giuliadangelo/workspace/data/DATASETs/CRIB/CRIB400/train_data/workingmemorybbox30050epochs/"
    
    # Alternative paths to try
    alternative_paths = [
        "workingmemorybbox30050epochs/",
        "working_memory/",
        "memory_output/",
        "./workingmemory/",
    ]
    
    print("Working Memory Analysis Pipeline")
    print("="*50)
    
    # Try to find memory directory
    memory_dir = None
    if os.path.exists(MEMORY_DIR):
        memory_dir = MEMORY_DIR
    else:
        print(f"Primary path not found: {MEMORY_DIR}")
        print("Trying alternative paths...")
        
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                memory_dir = alt_path
                print(f"Found alternative path: {alt_path}")
                break
        
        if memory_dir is None:
            print("❌ No working memory directory found!")
            print("Please ensure you have run Code2CreatingWorkingMemory.py first")
            print("Or update the MEMORY_DIR path in this script")
            return
    
    # Initialize analyzer
    analyzer = WorkingMemoryAnalyzer(memory_dir, max_objects=10)
    
    # Load working memories
    if not analyzer.load_working_memories():
        print("❌ Failed to load working memories")
        return
    
    # Analyze memory properties
    analyzer.analyze_memory_properties()
    
    # Create t-SNE visualization
    embeddings, labels = analyzer.visualize_with_tsne()
    
    # Compare with image features if available
    analyzer.compare_with_image_features()
    
    print("\n✅ Analysis complete!")
    print("Generated files:")
    print("  - working_memory_tsne_visualization.png")
    print("  - working_memory_analysis.json")
    if analyzer.features:
        print("  - memory_vs_features_comparison.png")


if __name__ == "__main__":
    main()