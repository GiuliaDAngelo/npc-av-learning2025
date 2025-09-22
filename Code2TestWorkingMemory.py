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
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from natsort import natsorted

class WorkingMemoryAnalyzer:
    """Analyzes and visualizes working memory representations"""
    
    def __init__(self, memory_dir, max_objects=10):
        self.memory_dir = memory_dir
        self.max_objects = max_objects
        self.individual_memories = {}  # obj_name -> {seq_label -> memory}
        self.combined_memories = {}    # obj_name -> combined_memory
        self.features = {}
        self.object_names = []
        self.output_dir = os.path.join(os.path.dirname(memory_dir), 'memory_analysis_plots')
        
    def load_working_memories(self):
        """Load working memory files from the specified directory"""
        if not os.path.exists(self.memory_dir):
            print(f"❌ Memory directory not found: {self.memory_dir}")
            return False
            
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Output plots will be saved to: {self.output_dir}")
            
        # Find memory files
        memory_files = [f for f in os.listdir(self.memory_dir) if f.endswith('_memory.npy')]
        
        if not memory_files:
            print(f"❌ No memory files found in {self.memory_dir}")
            return False
            
        print(f"Found {len(memory_files)} memory files")
        
        # Separate combined and individual memory files
        combined_files = [f for f in memory_files if '_combined_memory.npy' in f]
        individual_files = [f for f in memory_files if '_combined_memory.npy' not in f]
        
        # Load combined memories
        for memory_file in natsorted(combined_files):
            obj_name = memory_file.replace('_combined_memory.npy', '')
            memory_path = os.path.join(self.memory_dir, memory_file)
            
            try:
                memory = np.load(memory_path)
                self.combined_memories[obj_name] = memory
                if obj_name not in self.object_names:
                    self.object_names.append(obj_name)
                print(f"  ✅ Loaded combined memory for {obj_name}: shape {memory.shape}")
            except Exception as e:
                print(f"  ❌ Error loading combined memory for {obj_name}: {e}")
        
        # Load individual memories
        for memory_file in natsorted(individual_files):
            # Parse filename: obj_seq_memory.npy
            base_name = memory_file.replace('_memory.npy', '')
            parts = base_name.split('_')
            if len(parts) >= 2:
                # Try to separate object name and sequence
                seq_label = parts[-1]
                obj_name = '_'.join(parts[:-1])
            else:
                # Old format fallback
                obj_name = base_name
                seq_label = '0'
            
            memory_path = os.path.join(self.memory_dir, memory_file)
            
            try:
                memory = np.load(memory_path)
                
                if obj_name not in self.individual_memories:
                    self.individual_memories[obj_name] = {}
                self.individual_memories[obj_name][seq_label] = memory
                
                if obj_name not in self.object_names:
                    self.object_names.append(obj_name)
                    
                print(f"  ✅ Loaded individual memory for {obj_name}/{seq_label}: shape {memory.shape}")
            except Exception as e:
                print(f"  ❌ Error loading individual memory {memory_file}: {e}")
        
        # Sort object names
        self.object_names = natsorted(list(set(self.object_names)))
        
        print(f"Successfully loaded memories for {len(self.object_names)} objects")
        print(f"  Combined memories: {len(self.combined_memories)}")
        print(f"  Individual memories: {sum(len(seqs) for seqs in self.individual_memories.values())}")
        
        return len(self.object_names) > 0
    
    def visualize_with_tsne_combined(self, perplexity=5, random_state=42):
        """Create t-SNE visualization showing both individual and combined memories"""
        if not self.individual_memories and not self.combined_memories:
            print("❌ No memories loaded")
            return
            
        print("Creating t-SNE visualization with individual and combined memories...")
        
        # Prepare data
        memory_data = []
        labels = []
        point_types = []  # 'individual' or 'combined'
        colors_list = []
        
        # Color map for objects
        obj_colors = plt.cm.tab10(np.linspace(0, 1, len(self.object_names)))
        color_map = {obj: color for obj, color in zip(self.object_names, obj_colors)}
        
        # Add individual memories
        for obj_name in self.object_names:
            if obj_name in self.individual_memories:
                for seq_label, memory in self.individual_memories[obj_name].items():
                    memory_data.append(memory)
                    labels.append(f"{obj_name}_{seq_label}")
                    point_types.append('individual')
                    colors_list.append(color_map[obj_name])
        
        # Add combined memories
        for obj_name in self.object_names:
            if obj_name in self.combined_memories:
                memory = self.combined_memories[obj_name]
                memory_data.append(memory)
                labels.append(f"{obj_name}_avg")
                point_types.append('combined')
                colors_list.append(color_map[obj_name])
        
        if not memory_data:
            print("❌ No valid memory data to visualize")
            return
        
        memory_matrix = np.vstack(memory_data)
        print(f"Memory matrix shape: {memory_matrix.shape}")
        
        # Adjust perplexity
        n_samples = len(labels)
        perplexity = min(perplexity, max(1, n_samples - 1))
        
        # Apply t-SNE
        print("Running t-SNE...")
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, 
                   max_iter=1000, verbose=0)
        embeddings_2d = tsne.fit_transform(memory_matrix)
        
        # Create visualization
        plt.figure(figsize=(14, 10))
        
        # Plot individual memories as circles
        for i, (label, point_type, color) in enumerate(zip(labels, point_types, colors_list)):
            if point_type == 'individual':
                plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], 
                           c=[color], s=60, alpha=0.6, marker='o', 
                           label=f"{label.split('_')[0]} (seq)" if label.split('_')[0] not in [l.split(' ')[0] for l in plt.gca().get_legend_handles_labels()[1] if ' (seq)' in l] else "")
        
        # Plot combined memories as stars
        for i, (label, point_type, color) in enumerate(zip(labels, point_types, colors_list)):
            if point_type == 'combined':
                plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], 
                           c=[color], s=200, alpha=0.9, marker='*', 
                           edgecolors='black', linewidths=1,
                           label=f"{label.replace('_avg', '')} (avg)")
                
                # Add text labels for combined memories
                plt.annotate(label.replace('_avg', ''), 
                           (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                           xytext=(5, 5), textcoords='offset points', 
                           fontsize=10, fontweight='bold', alpha=0.8)
        
        plt.title('Working Memory Representations (t-SNE)\nCircles: Individual Sequences, Stars: Combined Averages')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(self.output_dir, 'working_memory_tsne_combined.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ t-SNE visualization saved to {output_path}")
        #plt.show()
        
        return embeddings_2d, labels, point_types
    
    def create_similarity_matrix_averages(self):
        """Create similarity matrix between average (combined) object memories"""
        if not self.combined_memories:
            print("❌ No combined memories available for similarity analysis")
            return
            
        print("Creating similarity matrix for average object memories...")
        
        # Prepare combined memory data
        obj_names = natsorted(list(self.combined_memories.keys()))
        memory_data = []
        
        for obj_name in obj_names:
            memory = self.combined_memories[obj_name]
            memory_data.append(memory)
        
        if len(memory_data) < 2:
            print("❌ Need at least 2 objects for similarity matrix")
            return
            
        memory_matrix = np.vstack(memory_data)
        
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(memory_matrix)
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix, 
                   xticklabels=obj_names, 
                   yticklabels=obj_names,
                   annot=True, 
                   fmt='.3f',
                   cmap='RdYlBu_r',
                   center=0,
                   square=True,
                   cbar_kws={'label': 'Cosine Similarity'})
        
        plt.title('Similarity Matrix: Average Object Memories\n(Combined across sequences)')
        plt.xlabel('Objects')
        plt.ylabel('Objects')
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(self.output_dir, 'similarity_matrix_averages.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Average similarity matrix saved to {output_path}")
        #plt.show()
        
        return similarity_matrix, obj_names
    
    def create_similarity_matrix_objects_vs_averages(self):
        """Create similarity matrix between individual object sequences and their averages"""
        if not self.individual_memories or not self.combined_memories:
            print("❌ Need both individual and combined memories for comparison")
            return
            
        print("Creating similarity matrix: objects vs their averages...")
        
        # Prepare data for objects that have both individual and combined memories
        common_objects = set(self.individual_memories.keys()) & set(self.combined_memories.keys())
        if not common_objects:
            print("❌ No objects with both individual and combined memories")
            return
            
        individual_data = []
        individual_labels = []
        combined_data = []
        combined_labels = []
        
        for obj_name in natsorted(common_objects):
            # Add combined memory
            combined_memory = self.combined_memories[obj_name]
            combined_data.append(combined_memory)
            combined_labels.append(f"{obj_name}_avg")
            
            # Add individual memories
            for seq_label, memory in self.individual_memories[obj_name].items():
                individual_data.append(memory)
                individual_labels.append(f"{obj_name}_{seq_label}")
        
        if not individual_data or not combined_data:
            print("❌ No data available for comparison")
            return
            
        individual_matrix = np.vstack(individual_data)
        combined_matrix = np.vstack(combined_data)
        
        # Calculate cross-similarity between individuals and averages
        cross_similarity = cosine_similarity(individual_matrix, combined_matrix)
        
        # Create heatmap
        plt.figure(figsize=(12, len(individual_labels) * 0.5 + 3))
        sns.heatmap(cross_similarity,
                   xticklabels=combined_labels,
                   yticklabels=individual_labels,
                   annot=True,
                   fmt='.3f',
                   cmap='RdYlBu_r',
                   center=0,
                   cbar_kws={'label': 'Cosine Similarity'})
        
        plt.title('Similarity Matrix: Individual Sequences vs Average Memories')
        plt.xlabel('Average Memories')
        plt.ylabel('Individual Sequences')
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(self.output_dir, 'similarity_matrix_objects_vs_averages.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Objects vs averages similarity matrix saved to {output_path}")
        #plt.show()
        
        return cross_similarity, individual_labels, combined_labels
    
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
            # Apply t-SNE
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, 
                       max_iter=1000, verbose=1)
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
            #plt.show()
            
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


def main():
    """Main analysis pipeline"""
    
    # Configuration - update these paths as needed
    ROOT = '/home/matt/DATA/CRIB/'
    MEMORY_DIR = ROOT + 'workingmemory_from_bboxes/'
    
    # Alternative paths to try
    alternative_paths = [
        "/home/matt/DATA/CRIB/workingmemorybbox30050epochs/",
        "./workingmemory_from_bboxes/",
        "./working_memory/",
        "./memory_output/",
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
            print("Please ensure you have run Code1CreateWorkingMemory.py first")
            print("Or update the MEMORY_DIR path in this script")
            return
    
    # Initialize analyzer
    analyzer = WorkingMemoryAnalyzer(memory_dir, max_objects=20)
    
    # Load working memories
    if not analyzer.load_working_memories():
        print("❌ Failed to load working memories")
        return
    
    print("\n" + "="*60)
    print("RUNNING VISUALIZATIONS AND ANALYSIS")
    print("="*60)
    
    # Create t-SNE visualization with combined memories
    analyzer.visualize_with_tsne_combined()
    
    # Create similarity matrix for average memories
    analyzer.create_similarity_matrix_averages()
    
    # Create similarity matrix: objects vs averages
    analyzer.create_similarity_matrix_objects_vs_averages()
    
    print("\n✅ Analysis complete!")
    print("Generated files in:", analyzer.output_dir)
    print("  - working_memory_tsne_combined.png")
    print("  - similarity_matrix_averages.png") 
    print("  - similarity_matrix_objects_vs_averages.png")


if __name__ == "__main__":
    main()
