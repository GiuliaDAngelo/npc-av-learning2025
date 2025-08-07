import numpy as np
from attention_helpers import initialise_attention, run_attention
from oms_helpers import initialize_oms, egomotion
import torch
import cv2
import os
from natsort import natsorted
from PIL import Image
import torchvision.transforms as T
import matplotlib

matplotlib.use('Agg')  # Use Agg backend (no GUI)
import matplotlib.pyplot as plt
from collections import deque

# Add the EmbeddingExtractor import at the top
from load_model import EmbeddingExtractor

import sspspace
import torchvision.transforms as transforms

# Enhanced transforms that match your training data
event_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # Match training size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Match training normalization
])

transform = T.Compose([
    T.Grayscale(),
    T.ToTensor(),
])


class WorkingMemoryManager:
    """Enhanced working memory management with quality-based filtering"""

    def __init__(self, memory_size=15, similarity_threshold=0.85, gamma=0.9, min_attention_threshold=100):
        self.memory_size = memory_size
        self.similarity_threshold = similarity_threshold
        self.gamma = gamma
        self.min_attention_threshold = min_attention_threshold

        # Memory buffers
        self.feature_buffer = deque(maxlen=memory_size)
        self.coord_buffer = deque(maxlen=memory_size)
        self.quality_scores = deque(maxlen=memory_size)
        self.frame_indices = deque(maxlen=memory_size)

        # Statistics
        self.total_frames = 0
        self.stored_frames = 0

    def calculate_feature_quality(self, roi_patch, attention_strength, patch_variance=None):
        """Calculate quality score for a feature based on multiple factors"""
        # Normalize attention strength (0-255 -> 0-1)
        attention_score = min(attention_strength / 255.0, 1.0)

        # Calculate patch informativeness
        if patch_variance is None:
            patch_variance = np.var(roi_patch)

        # Normalize variance score (higher variance = more informative)
        variance_score = min(patch_variance / 1000.0, 1.0)

        # Check if patch has sufficient detail (not too uniform)
        detail_score = 1.0 if patch_variance > 100 else 0.5

        # Combined quality score
        quality = (attention_score * 0.5 + variance_score * 0.3 + detail_score * 0.2)

        return quality

    def is_significantly_different(self, new_features, threshold=None):
        """Check if new features are significantly different from recent memory"""
        if not self.feature_buffer:
            return True

        threshold = threshold or self.similarity_threshold

        # Compare with recent features (last 3-5 frames)
        recent_count = min(5, len(self.feature_buffer))
        recent_features = list(self.feature_buffer)[-recent_count:]

        for recent_features_vec in recent_features:
            # Cosine similarity
            similarity = np.dot(new_features, recent_features_vec) / (
                    np.linalg.norm(new_features) * np.linalg.norm(recent_features_vec) + 1e-8
            )
            if similarity > threshold:
                return False
        return True

    def should_store_memory(self, roi_patch, attention_strength, new_features, frame_idx):
        """Determine if this frame should be stored in working memory"""
        self.total_frames += 1

        # Calculate quality metrics
        patch_variance = np.var(roi_patch)
        quality_score = self.calculate_feature_quality(roi_patch, attention_strength, patch_variance)

        # Multiple criteria for storage
        criteria = {
            'quality_threshold': quality_score > 0.4,
            'attention_threshold': attention_strength > self.min_attention_threshold,
            'novelty_check': self.is_significantly_different(new_features),
            'patch_variance': patch_variance > 50,  # Ensure patch isn't too uniform
        }

        # Store if most criteria are met
        should_store = sum(criteria.values()) >= 3

        if should_store:
            self.stored_frames += 1

        return should_store, quality_score, criteria

    def update_memory(self, features, coords, quality_score, frame_idx):
        """Update working memory with new high-quality features"""
        self.feature_buffer.append(features.copy())
        self.coord_buffer.append(coords.copy())
        self.quality_scores.append(quality_score)
        self.frame_indices.append(frame_idx)

    def get_consolidated_memory(self, coord_encoder):
        """Get consolidated memory representation using quality weighting"""
        if not self.feature_buffer:
            return coord_encoder.encode([[0, 0]])

        # Convert to arrays
        features_array = np.array(self.feature_buffer)
        coords_array = np.array(self.coord_buffer)
        weights = np.array(self.quality_scores)

        # Normalize weights
        weights = weights / np.sum(weights)

        # Weighted average of features and coordinates
        consolidated_features = np.average(features_array, axis=0, weights=weights)
        consolidated_coords = np.average(coords_array, axis=0, weights=weights)

        # Ensure features are normalized
        consolidated_features = consolidated_features / (np.linalg.norm(consolidated_features) + 1e-8)

        # Combine spatial and feature information using SSP
        roi_center = coord_encoder.encode([[consolidated_coords[0], consolidated_coords[1]]])
        img_feat_ssp = sspspace.SSP(consolidated_features)

        return roi_center * img_feat_ssp

    def get_statistics(self):
        """Get memory storage statistics"""
        storage_rate = (self.stored_frames / self.total_frames) if self.total_frames > 0 else 0
        return {
            'total_frames': self.total_frames,
            'stored_frames': self.stored_frames,
            'storage_rate': storage_rate,
            'buffer_size': len(self.feature_buffer),
            'avg_quality': np.mean(self.quality_scores) if self.quality_scores else 0
        }


class Config:
    MAX_X, MAX_Y = 128, 128

    OMS_PARAMS = {
        'size_krn_center': 8,
        'sigma_center': 1,
        'size_krn_surround': 8,
        'sigma_surround': 4,
        'threshold': 0.96,
        'tau_memOMS': 0.3,
        'sc': 1,
        'ss': 1
    }

    ATTENTION_PARAMS = {
        'size_krn': 16,
        'r0': 14,
        'rho': 0.05,
        'theta': np.pi * 3 / 2,
        'thetas': np.arange(0, 2 * np.pi, np.pi / 4),
        'thick': 3,
        'fltr_resize_perc': [2, 2],
        'offsetpxs': 0,
        'offset': (0, 0),
        'num_pyr': 6,
        'tau_mem': 0.3,
        'stride': 1,
        'out_ch': 1
    }


def load_trained_model(device="cpu"):
    """Load the trained autoencoder model via EmbeddingExtractor"""
    try:
        # Use the EmbeddingExtractor class which handles model loading
        model = EmbeddingExtractor(
            model_path="autoencoder-classifier-trained/model.pth",
            info_path="autoencoder-classifier-trained/training_info.json"
        )

        # Get training info
        class_names = model.training_info.get('class_names', [])
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}

        return model, class_names, class_to_idx, 'embedding_extractor'

    except Exception as e:
        print(f"Failed to load EmbeddingExtractor: {e}")
        return None, None, None, None


def process_object_sequence(obj, obj_path_data, model, coord_encoder, device, config,
                            net_center, net_surround, net_attention, bbox_saving_path):
    """Enhanced object sequence processing with selective memory storage"""

    # Initialize enhanced memory manager
    memory_manager = WorkingMemoryManager(
        memory_size=20,
        similarity_threshold=0.80,
        min_attention_threshold=80
    )

    # Get data files
    data_files = natsorted([f for f in os.listdir(obj_path_data)
                            if os.path.isfile(os.path.join(obj_path_data, f)) and f != '.DS_Store'])

    print(f"Processing {len(data_files)} frames for object: {obj}")

    # Initialize variables
    max_x, max_y = 400, 400
    resolution = (max_y, max_x)
    box_size = 350
    size_krn_after_oms = 343
    OMS = np.zeros((size_krn_after_oms, size_krn_after_oms), dtype=np.float32)
    vSliceOMS = torch.zeros((1, size_krn_after_oms, size_krn_after_oms), dtype=torch.float32).to(device)
    saliency_map = np.zeros((max_y, max_x), dtype=np.float32)
    salmax_coords = np.zeros((2,), dtype=np.int32)

    # Process each frame
    for frame_idx, data_file_i in enumerate(data_files):
        img_path = os.path.join(obj_path_data, data_file_i)

        try:
            # Load and preprocess image
            img = Image.open(img_path)
            window = transform(img)
            window_original = window.clone()

            # Computing egomotion
            wOMS = torch.tensor(window, dtype=torch.float32).to(device)
            OMS, indexes = egomotion(wOMS, net_center, net_surround, device, config.MAX_Y,
                                     config.MAX_X, config.OMS_PARAMS['threshold'])

            vSliceOMS = OMS.squeeze(0)

            # Run attention mechanism
            saliency_map[:], salmax_coords[:] = run_attention(
                window, net_attention, device, resolution, config.ATTENTION_PARAMS['num_pyr']
            )

            # Convert window to displayable format
            window_img = window.detach().cpu().numpy().squeeze(0)
            window_img = (window_img * 255).clip(0, 255).astype(np.uint8)
            window_img_color = cv2.cvtColor(window_img, cv2.COLOR_GRAY2BGR)

            # Extract ROI based on attention
            x, y = salmax_coords[1], salmax_coords[0]
            x1, y1 = max(x - box_size // 2, 0), max(y - box_size // 2, 0)
            x2, y2 = min(x + box_size // 2, window_img.shape[1]), min(y + box_size // 2, window_img.shape[0])

            roi_crop = window_img_color[y1:y2, x1:x2]
            attention_strength = saliency_map[y, x]

            # Convert to RGB for model consistency
            roi_crop_rgb = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2RGB)
            roi_tensor = event_transform(roi_crop_rgb).unsqueeze(0).to(device)

            # Extract features using EmbeddingExtractor
            with torch.no_grad():
                # Convert ROI to proper format for EmbeddingExtractor
                # EmbeddingExtractor expects (H, W, 3) numpy array with values 0-255
                roi_crop_uint8 = roi_crop_rgb.astype(np.uint8)

                # Get embeddings using the EmbeddingExtractor
                image_features = model.get_embeddings(roi_crop_uint8)

                # Ensure features are 1D and normalized
                if len(image_features.shape) > 1:
                    image_features = image_features.flatten()
                image_features = image_features / (np.linalg.norm(image_features) + 1e-8)

            # Check if we should store this frame in working memory
            should_store, quality_score, criteria = memory_manager.should_store_memory(
                roi_crop_rgb, attention_strength, image_features, frame_idx
            )

            if should_store:
                memory_manager.update_memory(image_features, [x, y], quality_score, frame_idx)
                print(f"Frame {frame_idx:3d}: Stored (quality: {quality_score:.3f}, "
                      f"attention: {attention_strength:6.1f}) - {criteria}")

            # Optional: Save bounding box images
            if bbox_saving_path and os.path.exists(bbox_saving_path):
                name_file = data_file_i.split('.png')[0]
                save_path = os.path.join(bbox_saving_path, obj, f"{name_file}_bbox.png")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                save_img = np.zeros((box_size, box_size), dtype=np.uint8)
                crop = window_img[y1:y2, x1:x2]
                h, w = crop.shape[:2]
                save_img[:h, :w] = crop
                cv2.imwrite(save_path, save_img)

            # Visualization (optional)
            if frame_idx % 50 == 0:  # Show every 50th frame
                window_img_colorized = cv2.cvtColor(window_img, cv2.COLOR_GRAY2BGR)
                cv2.rectangle(window_img_colorized, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Could add visualization here if needed
                # cv2.imshow(f'Processing {obj}', window_img_colorized)
                # cv2.waitKey(1)

        except Exception as e:
            print(f"Error processing frame {frame_idx} ({data_file_i}): {e}")
            continue

        finally:
            # Clean up memory
            del window
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Reset for next frame
            window = torch.zeros((1, max_y, max_x), dtype=torch.float32)
            saliency_map = np.zeros((max_y, max_x), dtype=np.float32)

    # Get final consolidated memory
    final_memory = memory_manager.get_consolidated_memory(coord_encoder)
    stats = memory_manager.get_statistics()

    print(f"Object {obj} complete: {stats}")

    return final_memory, memory_manager.feature_buffer, stats


def main():
    """Enhanced main processing function"""

    # Device setup
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
    else "cpu")
    print(f"Using device: {device}")

    # Configuration
    config = Config()

    # Paths
    root = '/Users/giuliadangelo/Downloads/npc-av-learning/CRIB/train_data/'
    path_data = root + 'bbox/'
    bbox_saving_path = root + 'bbox/'
    memory_save_path = '/Users/giuliadangelo/Downloads/npc-av-learning/CRIB/workingmemory/'

    # Create directories
    os.makedirs(memory_save_path, exist_ok=True)

    # Load trained model using EmbeddingExtractor
    try:
        model, class_names, class_to_idx, model_type = load_trained_model(device=device)
        if model is not None:
            print(f"✅ Loaded {model_type} successfully!")
            if class_names:
                print(f"Model trained on {len(class_names)} classes: {class_names}")
            else:
                print("No class names found in training info")
        else:
            raise Exception("Could not load model")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Make sure the model files exist:")
        print("  - autoencoder-classifier-trained/model.pth")
        print("  - autoencoder-classifier-trained/training_info.json")
        return

    # Initialize networks
    net_center, net_surround = initialize_oms(device, config.OMS_PARAMS)
    net_attention = initialise_attention(device, config.ATTENTION_PARAMS)

    # Initialize coordinate encoder
    coord_encoder = sspspace.RandomSSPSpace(domain_dim=2, ssp_dim=512)

    # Get objects to process
    objects = natsorted([d for d in os.listdir(path_data)
                         if os.path.isdir(os.path.join(path_data, d))])

    print(f"Found {len(objects)} objects to process: {objects}")

    # Process each object
    results_summary = {}

    for obj_idx, obj in enumerate(objects):
        print(f"\n{'=' * 60}")
        print(f"Processing object {obj_idx + 1}/{len(objects)}: {obj}")
        print(f"{'=' * 60}")

        obj_path_data = os.path.join(path_data, obj)

        try:
            # Process the object sequence
            final_memory, stored_features, stats = process_object_sequence(
                obj, obj_path_data, model, coord_encoder, device, config,
                net_center, net_surround, net_attention, bbox_saving_path
            )

            # Save results
            memory_file = os.path.join(memory_save_path, f'{obj}_memory.npy')
            features_file = os.path.join(memory_save_path, f'{obj}_features_sequence.npy')

            np.save(memory_file, final_memory)

            if stored_features:
                np.save(features_file, np.array(stored_features))

            # Store results
            results_summary[obj] = stats

            print(f"✅ Saved memory for {obj}: {stats['stored_frames']} high-quality frames stored")

        except Exception as e:
            print(f"❌ Error processing object {obj}: {e}")
            results_summary[obj] = {'error': str(e)}

    # Print final summary
    print(f"\n{'=' * 60}")
    print("PROCESSING COMPLETE - SUMMARY")
    print(f"{'=' * 60}")

    total_frames = sum(stats.get('total_frames', 0) for stats in results_summary.values() if 'total_frames' in stats)
    total_stored = sum(stats.get('stored_frames', 0) for stats in results_summary.values() if 'stored_frames' in stats)

    for obj, stats in results_summary.items():
        if 'error' in stats:
            print(f"{obj:20s}: ERROR - {stats['error']}")
        else:
            print(f"{obj:20s}: {stats['stored_frames']:3d}/{stats['total_frames']:3d} frames "
                  f"({stats['storage_rate']:.1%}) - Avg quality: {stats['avg_quality']:.3f}")

    print(f"\nOverall: {total_stored}/{total_frames} frames stored ({total_stored / total_frames:.1%})")

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()





# import numpy as np
# from attention_helpers import initialise_attention, run_attention
# from oms_helpers import initialize_oms, egomotion
# import torch
# import cv2
# import os
# from natsort import natsorted
# from PIL import Image
# import torchvision.transforms as T
# import matplotlib
# matplotlib.use('Agg')  # Use Agg backend (no GUI)
# import matplotlib.pyplot as plt
#
# from load_model import EmbeddingExtractor
#
# import sspspace
# from transformers import CLIPProcessor, CLIPModel
#
# import torchvision.transforms as transforms
#
# event_transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Grayscale(num_output_channels=3),
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5]*3, [0.5]*3)
# ])
#
#
# transform = T.Compose([
#     T.Grayscale(),
#     T.ToTensor(),
# ])
#
# class Config:
#     MAX_X, MAX_Y = 128, 128
#
#     OMS_PARAMS = {
#         'size_krn_center': 8,
#         'sigma_center': 1,
#         'size_krn_surround': 8,
#         'sigma_surround': 4,
#         'threshold': 0.96,
#         'tau_memOMS': 0.3,
#         'sc': 1,
#         'ss': 1
#     }
#
#     ATTENTION_PARAMS = {
#         'size_krn': 16,
#         'r0': 14,
#         'rho': 0.05,
#         'theta': np.pi * 3 / 2,
#         'thetas': np.arange(0, 2 * np.pi, np.pi / 4),
#         'thick': 3,
#         'fltr_resize_perc': [2, 2],
#         'offsetpxs': 0,
#         'offset': (0, 0),
#         'num_pyr': 6,
#         'tau_mem': 0.3,
#         'stride': 1,
#         'out_ch': 1
#     }
#
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# config = Config()
#
#
# bbox_savingFLAG = False
# root = '/Users/giuliadangelo/Downloads/npc-av-learning/CRIB/train_data/'
# path_data = root+'bbox/'
# objects = natsorted([d for d in os.listdir(path_data) if os.path.isdir(os.path.join(path_data, d))])
#
# for obj in objects:
#     print(obj)
#     #sensor
#     max_x, max_y = 400, 400
#     resolution = (max_y, max_x)
#     box_size = 350
#     #oms
#     size_krn_after_oms = 343
#     OMS = np.zeros((size_krn_after_oms, size_krn_after_oms), dtype=np.float32)
#     vSliceOMS = torch.zeros((1, size_krn_after_oms, size_krn_after_oms), dtype=torch.float32).to(device)
#     #saliency map
#     saliency_map = np.zeros((max_y, max_x), dtype=np.float32)
#     salmax_coords = np.zeros((2,), dtype=np.int32)
#     #encoder
#     coord_encoder = sspspace.RandomSSPSpace(domain_dim=2, ssp_dim=512)
#
#     # clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
#     # clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
#     # clip_model.eval()
#
#     NUM_CLASSES = 17  # Set to number of categories you trained on
#     model_path = "clip_event_classifier.pth"
#
#     model = EmbeddingExtractor()
#
#     net_center, net_surround = initialize_oms(device, config.OMS_PARAMS)
#     net_attention = initialise_attention(device, config.ATTENTION_PARAMS)
#     obj_path_data = os.path.join(path_data, obj)
#
#     object_memory = coord_encoder.encode([[0, 0]])
#     bbox_saving_path = root + 'bbox/'
#     os.mkdir(bbox_saving_path+obj) if not os.path.exists(bbox_saving_path + obj) else None
#     data_files = natsorted([f for f in os.listdir(obj_path_data) if os.path.isfile(os.path.join(obj_path_data, f)) and f != '.DS_Store'])
#
#     for data_file_i in data_files:
#         img_path = os.path.join(obj_path_data, data_file_i)
#
#         img = Image.open(img_path)
#         window = transform(img)
#         window_original = window
#
#         # computing egomotion
#         wOMS = torch.tensor(window, dtype=torch.float32).to(device)
#         OMS, indexes = egomotion(wOMS, net_center, net_surround, device, config.MAX_Y,
#                                  config.MAX_X, config.OMS_PARAMS['threshold'])
#
#
#
#         vSliceOMS = OMS.squeeze(0)
#         #window
#         saliency_map[:], salmax_coords[:] = run_attention(
#             window, net_attention, device, resolution, config.ATTENTION_PARAMS['num_pyr']
#         )
#         # #OMS
#         # saliency_map[:], salmax_coords[:] = run_attention(
#         #     vSliceOMS[:], net_attention, device, resolution, config.ATTENTION_PARAMS['num_pyr']
#         # )
#
#         fig, ax = plt.subplots(figsize=(4, 4))
#         cax = ax.imshow(saliency_map, cmap='jet', vmin=0, vmax=255)
#         ax.scatter(salmax_coords[1], salmax_coords[0], s=60, c='white', edgecolors='black', linewidths=1)
#         # ax.set_title('Saliency')
#         ax.axis('off')
#         fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04, label='Saliency (0–255)')
#
#         fig.canvas.draw()
#         plt.close(fig)
#
#         # --- Events image with bounding box (right) ---
#         window_img = window.detach().cpu().numpy().squeeze(0)
#         window_img = (window_img * 255).clip(0, 255).astype(np.uint8)
#         window_img_color = cv2.cvtColor(window_img, cv2.COLOR_GRAY2BGR)
#
#
#         window_original = window_original.detach().cpu().numpy().squeeze(0)
#         window_original = (window_original * 255).clip(0, 255).astype(np.uint8)
#         window_original_colour = cv2.cvtColor(window_original, cv2.COLOR_GRAY2BGR)
#
#
#         x, y = salmax_coords[1], salmax_coords[0]
#         x1, y1 = max(x - box_size//2, 0), max(y - box_size//2, 0)
#         x2, y2 = min(x + box_size//2, window_img.shape[1]), min(y + box_size//2, window_img.shape[0])
#
#         cv2.rectangle(window_img_color, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         # cv2.putText(window_img_color, 'Events', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
#
#         mask = np.zeros_like(window_img_color)
#         mask[y1:y2, x1:x2] = window_img_color[y1:y2, x1:x2]
#         window_img_boxed = mask
#
#         roi_center = coord_encoder.encode([[x, y]])
#
#         if bbox_savingFLAG:
#             name_file = data_file_i.split('.png')[0]
#             save_path = os.path.join(bbox_saving_path+obj, f"{name_file}_bbox.png")
#             save_img = np.zeros((box_size, box_size), dtype=np.uint8)
#             crop = window_img[y1:y2, x1:x2]
#             h, w = crop.shape[:2]
#             save_img[:h, :w] = crop
#             cv2.imwrite(save_path, save_img)
#
#         # current_roi = clip_processor(images=[window_img_color[y1:y2, x1:x2]], return_tensors="pt", padding=True).to(device)
#         # with torch.no_grad():
#         #     image_features = clip_model.get_image_features(**current_roi)
#
#         roi_crop = window_img_color[y1:y2, x1:x2]
#         roi_tensor = event_transform(roi_crop).unsqueeze(0).to(device)
#
#         with torch.no_grad():
#             image_features = model.get_embeddings(roi_tensor.detach().cpu().numpy().squeeze(0).transpose(1,2,0))
#             image_features = image_features / np.linalg.norm(image_features, axis=-1, keepdims=True)  # L2-normalize
#
#         img_feat_ssp = sspspace.SSP(image_features)
#         new_roi = roi_center * img_feat_ssp
#         gamma = 0.99 # 0.99 for bbox
#         object_memory = gamma * object_memory + (1 - gamma) * new_roi
#
#
#         # --- VISUALISATION --- #
#         window_img_colorized = cv2.cvtColor(window_img, cv2.COLOR_GRAY2BGR)
#         OMS = OMS.squeeze(0).squeeze(0).cpu().detach().numpy()
#         OMS_map_color = cv2.applyColorMap(OMS.astype(np.uint8), cv2.COLORMAP_JET)
#         saliency_map_color = cv2.applyColorMap(saliency_map.astype(np.uint8), cv2.COLORMAP_JET)
#
#         #resize images to match dimensions
#         target_height = saliency_map_color.shape[0]
#         target_width = saliency_map_color.shape[1]
#         OMS_map_color_resized = cv2.resize(OMS_map_color, (target_width, target_height))
#         window_img_boxed_resized = cv2.resize(window_img_boxed, (target_width, target_height))
#
#         # combined = np.hstack((window_original_colour, window_img_colorized, OMS_map_color_resized, saliency_map_color, window_img_boxed_resized))
#
#         # cv2.imshow('Window | OMS | Saliency | Atention', combined)
#         # cv2.waitKey(1)
#
#         # Free memory
#         del window
#         torch.mps.empty_cache()
#
#         # Reset
#         window = torch.zeros((1, max_y, max_x), dtype=torch.float32)
#         saliency_map = np.zeros((max_y, max_x), dtype=np.float32)
#
#     np.save(
#         f'/Users/giuliadangelo/Downloads/npc-av-learning/CRIB/workingmemory/{obj}_image_features.npy',
#         image_features)
#     np.save(
#         f'/Users/giuliadangelo/Downloads/npc-av-learning/CRIB/workingmemory/{obj}_memory.npy',
#         object_memory)
# cv2.destroyAllWindows()