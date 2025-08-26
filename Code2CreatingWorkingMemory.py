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

# Add missing imports for the model
import torch.nn as nn
import torch.nn.functional as F

# Add the EmbeddingExtractor import - REMOVED, using direct model loading
from load_model import EmbeddingExtractor

import sspspace
import torchvision.transforms as transforms

# FIXED: Transform to match your EVENT FRAME training (single channel grayscale)
event_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),  # Single channel grayscale (not 3-channel)
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Single channel normalization like your training
])

transform = T.Compose([
    T.Grayscale(),
    T.ToTensor(),
])


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
        # UPDATED: Use your actual model paths
        model_path = "/Users/giuliadangelo/workspace/data/DATASETs/CRIB/CRIB400/train_data/resultsbbox30050epochs/autoencoder-trained/model.pth"
        info_path = "/Users/giuliadangelo/workspace/data/DATASETs/CRIB/CRIB400/train_data/resultsbbox30050epochs/autoencoder-trained/training_info.json"

        model = EmbeddingExtractor(
            model_path=model_path,
            info_path=info_path
        )

        print(f"✅ Loaded EmbeddingExtractor successfully!")
        return model

    except Exception as e:
        print(f"Failed to load EmbeddingExtractor: {e}")
        print("Make sure these files exist:")
        print(f"  - {model_path}")
        print(f"  - {info_path}")
        return None


def main():
    """Recreate your original pipeline exactly - process evframes with attention to get proper coordinates"""

    # Device setup (exactly like your original)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    config = Config()
    OMSFLAG = False

    # Load trained model
    model = load_trained_model(device=device)
    if model is None:
        return

    # Paths (no need to save bboxes anymore)
    root = '/Users/giuliadangelo/workspace/data/DATASETs/CRIB/CRIB400/train_data/'
    path_data = root + 'evframes/'  # Use original event frames
    memory_save_path = root +'workingmemorytest/'#'workingmemorybbox30050epochs/'

    # Create directories
    os.makedirs(memory_save_path, exist_ok=True)

    # Initialize networks (exactly like your original)
    if OMSFLAG:
        net_center, net_surround = initialize_oms(device, config.OMS_PARAMS)
    net_attention = initialise_attention(device, config.ATTENTION_PARAMS)

    # Get objects (exactly like your original)
    objects = natsorted([d for d in os.listdir(path_data)
                         if os.path.isdir(os.path.join(path_data, d))])

    print(f"Found {len(objects)} objects to process: {objects}")

    # Process each object (exactly like your original loop)
    for obj in objects:
        print(f"\nProcessing object: {obj}")

        # Initialize variables (exactly like your original)
        max_x, max_y = 400, 400
        resolution = (max_y, max_x)
        box_size = 350
        if OMSFLAG:
            size_krn_after_oms = 343
            OMS = np.zeros((size_krn_after_oms, size_krn_after_oms), dtype=np.float32)
            vSliceOMS = torch.zeros((1, size_krn_after_oms, size_krn_after_oms), dtype=torch.float32).to(device)
        saliency_map = np.zeros((max_y, max_x), dtype=np.float32)
        salmax_coords = np.zeros((2,), dtype=np.int32)

        # Initialize coordinate encoder (exactly like your original) - spatial semantic pointer
        coord_encoder = sspspace.RandomSSPSpace(domain_dim=2, ssp_dim=512)

        # Initialize object memory (exactly like your original)
        object_memory = coord_encoder.encode([[0, 0]])

        # Get path to object's event frames
        obj_path_data = os.path.join(path_data, obj)

        # Get all event frame files (exactly like your original)
        data_files = natsorted([f for f in os.listdir(obj_path_data)
                                if os.path.isfile(os.path.join(obj_path_data, f)) and f != '.DS_Store'])

        print(f"  Processing {len(data_files)} event frames...")

        # Process each event frame (exactly like your original)
        for data_file_i in data_files:
            img_path = os.path.join(obj_path_data, data_file_i)

            try:
                # Load and preprocess image (exactly like your original)
                img = Image.open(img_path)
                window = transform(img)
                window_original = window

                # # Computing egomotion (exactly like your original)
                # wOMS = torch.tensor(window, dtype=torch.float32).to(device)
                # OMS, indexes = egomotion(wOMS, net_center, net_surround, device, config.MAX_Y,
                #                          config.MAX_X, config.OMS_PARAMS['threshold'])
                #
                # # Dynamically get the actual OMS size (no more hardcoding!)
                # if vSliceOMS is None:
                #     # Initialize vSliceOMS based on actual OMS output size
                #     oms_shape = OMS.shape
                #     print(f"    Detected OMS output shape: {oms_shape}")
                #     if len(oms_shape) == 4:  # [batch, channel, height, width]
                #         vSliceOMS = torch.zeros((1, oms_shape[2], oms_shape[3]), dtype=torch.float32).to(device)
                #     elif len(oms_shape) == 3:  # [channel, height, width]
                #         vSliceOMS = torch.zeros((1, oms_shape[1], oms_shape[2]), dtype=torch.float32).to(device)
                #     else:
                #         print(f"    Unexpected OMS shape: {oms_shape}")
                #         vSliceOMS = torch.zeros_like(OMS.squeeze(0) if len(oms_shape) > 3 else OMS).to(device)
                #
                # vSliceOMS = OMS.squeeze(0)

                # Run attention mechanism (exactly like your original)
                saliency_map[:], salmax_coords[:] = run_attention(
                    window, net_attention, device, resolution, config.ATTENTION_PARAMS['num_pyr']
                )

                # Convert window to displayable format (exactly like your original)
                window_img = window.detach().cpu().numpy().squeeze(0)
                window_img = (window_img * 255).clip(0, 255).astype(np.uint8)
                window_img_color = cv2.cvtColor(window_img, cv2.COLOR_GRAY2BGR)

                # Get coordinates (exactly like your original)
                x, y = salmax_coords[1], salmax_coords[0]
                x1, y1 = max(x - box_size // 2, 0), max(y - box_size // 2, 0)
                x2, y2 = min(x + box_size // 2, window_img.shape[1]), min(y + box_size // 2, window_img.shape[0])

                # Create ROI center encoding (exactly like your original)
                roi_center = coord_encoder.encode([[x, y]])

                # Extract features from ROI (updated to use your trained model with proper grayscale)
                roi_crop_gray = window_img[y1:y2, x1:x2]  # Keep as grayscale
                roi_tensor = event_transform(roi_crop_gray)

                ######## we have a mismatch of the size


                with torch.no_grad():
                    # Convert tensor to numpy format expected by EmbeddingExtractor
                    roi_for_model = roi_tensor.detach().cpu().numpy()
                    #
                    # # Denormalize from [-1,1] to [0,255] and convert to uint8
                    # roi_for_model = ((roi_for_model + 1) * 127.5).astype(np.uint8)

                    print("ROI shape", roi_for_model.shape)

                    # Get embeddings using your trained model
                    image_features = model.get_embeddings(roi_for_model)

                    # Ensure features are 1D and normalized
                    if len(image_features.shape) > 1:
                        image_features = image_features.flatten()
                    image_features = image_features / (np.linalg.norm(image_features) + 1e-8)

                # Create SSP and update memory (exactly like your original)
                img_feat_ssp = sspspace.SSP(image_features)
                new_roi = roi_center * img_feat_ssp
                gamma = 0.99  # Same as your original
                object_memory = gamma * object_memory + (1 - gamma) * new_roi

            except Exception as e:
                print(f"    Error processing {data_file_i}: {e}")
                continue

            finally:
                # Clean up memory (exactly like your original)
                del window
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                elif torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Reset for next frame (exactly like your original)
                window = torch.zeros((1, max_y, max_x), dtype=torch.float32)
                saliency_map = np.zeros((max_y, max_x), dtype=np.float32)

        # Save results (exactly like your original)
        print(f"  Saving memory for {obj}...")

        # Save the final consolidated memory (exactly like your original)
        memory_file = os.path.join(memory_save_path, f'{obj}_memory.npy')
        np.save(memory_file, object_memory)

        # Save the final image features (exactly like your original)
        features_file = os.path.join(memory_save_path, f'{obj}_image_features.npy')
        np.save(features_file, image_features)

        print(f"  ✅ Saved {obj} memory and features with proper coordinates")

    print(f"\n✅ All objects processed successfully!")
    print(f"Working memory saved to: {memory_save_path}")


if __name__ == '__main__':
    main()

    #### check from here you need to understand if the model I am loading is the one I trained before