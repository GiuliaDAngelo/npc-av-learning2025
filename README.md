# npc-av-learning2025
Object learning through saccades using contrastive deep Q-learning.

## Installation

1. Install required packages:
```bash
git clone git@github.com:ctn-waterloo/sspspace.git
cd sspspace
pip install . # to install sspspace

# use pip or conda to install
pip install torch torchvision matplotlib numpy scikit-learn
pip install opencv-python
pip install umap-learn # Optional, for UMAP analysis
pip install mujoco
pip install scikit-image
```

2. Download CRIB Data and unzip it to `./CRIB Data/`

## Training (`train.py`)

The training script implements contrastive deep Q-learning where an RL agent learns to control camera rotations to maximize similarity between instances of the same object class while minimizing similarity to different classes.

### Contrastive Learning Objective

For each episode with target object from class c:

**Reward = Δs⁺ - Δs⁻**

where:
- Δs⁺ = increase in cosine similarity to same-class instances
- Δs⁻ = increase in cosine similarity to different-class instances

The agent learns actions that bring same-class memories closer together while pushing different-class memories apart in the embedding space.

### Training Mode

Model is saved to ```./agent.pt```

```bash
python train.py --mode train --objects dog cat apple bee --num_episodes 20 --lr 0.001
```

**Key Parameters:**
- `--mode train`: Training mode with RL agent
- `--objects`: List of object names (without underscores)
- `--num_episodes`: Number of training episodes (randomly samples from objects)
- `--lr`: Learning rate for RL agent (default: 0.001)
- `--gamma`: Discount factor (default: 0.99)
- `--epsilon_start`: Initial exploration rate (default: 1.0)
- `--epsilon_end`: Final exploration rate (default: 0.01)
- `--epsilon_decay`: Exploration decay rate (default: 0.995)
- `--ssp_dim`: SSP vector dimension (default: 2500, use 1000 on Mac for memory)
- `--max_steps`: Steps per episode (default: 500)

**Memory Management (Mac):**
- Reduce `--ssp_dim 1000` if running out of memory
- Script automatically uses MPS (Metal) for GPU acceleration on Mac
- BATCH_SIZE=8 and REPLAY_BUFFER_SIZE=5000 for Mac compatibility

**Output:**
- Saves final policy: `discrimination_policy.pth`
- Saves memories with class info: `object_memories_TIMESTAMP.json`

### Inference Mode

Use a learned policy to build memories of new objects:

```bash
python train.py --mode inference --policy discrimination_policy.pth --objects dog cat bee --num_episodes 10
```

**Parameters:**
- `--mode inference`: Inference mode (no training)
- `--policy`: Path to saved policy file
- `--objects`: Objects to memorize
- `--num_episodes`: Number of examples to generate (randomly samples)

**Output:**
- Saves memories: `object_memories_inference_TIMESTAMP.json`

### Demo Mode

Manually explore objects with keyboard controls:

```bash
python train.py --mode demo --objects dog
```

**Controls:**
- ↑/↓: Increase/decrease pitch speed
- ←/→: Increase/decrease yaw speed
- ESC/Q: Quit

Current speeds displayed on screen. Shows detected saccades and ROI in real-time.

## Memory Analysis (`train_analysis.py`)

Visualize learned object memories using dimensionality reduction.

### Usage

```bash
python train_analysis.py object_memories_TIMESTAMP.json --method both --output_dir analysis
```

**Parameters:**
- First argument: Path to memories JSON file
- `--method`: Visualization method (`tsne`, `pca`, `both`) (default: `both`)
- `--output_dir`: Output directory for plots (default: `analysis`)

**Output:**
- `similarity_heatmap.png`: Cosine similarity matrix (sorted by class, with class boundaries)
- `pca_projection.png`: PCA 2D projection (color-coded by class)
- `tsne_projection.png`: t-SNE 2D projection (color-coded by class)
- `umap_projection.png`: UMAP 2D projection (if umap-learn installed)

### Interpreting Results

Good contrastive learning should show:
- **Similarity heatmap**: High values (bright) within class blocks, low values (dark) between classes
- **PCA/t-SNE/UMAP**: Tight clusters per class, well-separated between classes

## Example Workflow

```bash
# 1. Train on multiple object classes
python train.py --mode train --objects dog cat apple bee banana --num_episodes 50 --ssp_dim 1000

# 2. Analyze learned memories
python train_analysis.py object_memories_*.json --method both

# 3. Test on inference with learned policy
python train.py --mode inference --policy discrimination_policy.pth --objects dog cat --num_episodes 20

# 4. Analyze inference memories
python train_analysis.py object_memories_inference_*.json --method both
```

## Technical Details

- **RL Agent**: Deep Q-Network with experience replay and target network
- **Feature Extraction**: DINOv2 ViT (384-dim patch embeddings)
- **Memory Binding**: SSP (Spatial Semantic Pointers) for binding saccade coordinates, rotation quaternions, and visual features
- **Attention**: Bottom-up saliency-based saccade selection
- **Event Simulation**: DVS events via IEBCS (requires minimum object motion)
- **Device**: Automatic MPS (Mac) > CUDA > CPU selection for RL agent

