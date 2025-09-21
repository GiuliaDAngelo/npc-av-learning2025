"""
Event Frame Processing Pipeline
Processes event frames using attention mechanisms and extracts features using trained autoencoder
"""

import os
import numpy as np
import torch
import cv2
from PIL import Image
import torchvision.transforms as T
import matplotlib
from natsort import natsorted
import sspspace

# Local imports
from attention_helpers import initialise_attention, run_attention
from oms_helpers import initialize_oms, egomotion
from load_model import EmbeddingExtractor

matplotlib.use('Agg')  # No GUI backend


# Configuration paths
MODEL_PATH = "./autoencoder/patches_conv/best_model.ckpt"

ROOT = '/home/matt/DATA/CRIB/'
PATH_DATA = ROOT + 'train_event_frames/'
MEMORY_SAVE_PATH = ROOT + 'workingmemorybbox30050epochs/'


class Config:
    """Configuration parameters for the processing pipeline"""

    MAX_X, MAX_Y = 128, 128
    BOX_SIZE = 350
    GAMMA = 0.99  # Memory update factor

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


class EventFrameProcessor:
    """Main processor for event frames with attention and memory encoding"""

    def __init__(self, model_path, device=None):
        self.config = Config()

        # Setup device
        if device is None:
            self.device = torch.device(
                "mps" if torch.backends.mps.is_available()
                else "cuda" if torch.cuda.is_available()
                else "cpu"
            )
        else:
            self.device = device

        print(f"Using device: {self.device}")

        # Load model
        self.model = self._load_model(model_path)

        # Initialize networks
        self.net_attention = initialise_attention(self.device, self.config.ATTENTION_PARAMS)

        # Initialize transforms
        self.transform = T.Compose([
            T.Grayscale(),
            T.ToTensor(),
        ])

        # Initialize coordinate encoder
        self.coord_encoder = sspspace.RandomSSPSpace(domain_dim=2, ssp_dim=512)

    def _load_model(self, model_path):
        """Load the trained autoencoder model"""
        try:
            model = EmbeddingExtractor(
                model_path=model_path
            )
            print(f"✅ Model loaded successfully: {type(model).__name__}")
            return model
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            print(f"Model path: {model_path}")

    def process_frame(self, img_path, saliency_map, resolution):
        """Process a single event frame and return attention coordinates and features"""
        try:
            # Load and preprocess image
            img = Image.open(img_path)
            window = self.transform(img)

            # Run attention mechanism
            saliency_map[:], salmax_coords = run_attention(
                window, self.net_attention, self.device,
                resolution, self.config.ATTENTION_PARAMS['num_pyr']
            )

            # Convert to displayable format
            window_img = window.detach().cpu().numpy().squeeze(0)
            window_img = (window_img * 255).clip(0, 255).astype(np.uint8)

            # Extract ROI coordinates
            x, y = salmax_coords[1], salmax_coords[0]
            x1 = max(x - self.config.BOX_SIZE // 2, 0)
            y1 = max(y - self.config.BOX_SIZE // 2, 0)
            x2 = min(x + self.config.BOX_SIZE // 2, window_img.shape[1])
            y2 = min(y + self.config.BOX_SIZE // 2, window_img.shape[0])

            # Extract ROI and get features
            roi_crop = window_img[y1:y2, x1:x2]

            with torch.no_grad():
                image_features = self.model.get_embeddings(roi_crop)

            # Create spatial encoding
            roi_center = self.coord_encoder.encode([[x, y]])
            img_feat_ssp = sspspace.SSP(image_features)
            new_roi = roi_center * img_feat_ssp

            return new_roi, image_features, (x, y)

        except Exception as e:
            print(f"    Error processing frame {os.path.basename(img_path)}: {e}")
            return None, None, None

    def process_object(self, obj_path, obj_name):
        """Process all frames for a single object"""
        print(f"\nProcessing object: {obj_name}")

        # Setup processing variables
        max_x, max_y = 400, 400
        resolution = (max_y, max_x)
        saliency_map = np.zeros((max_y, max_x), dtype=np.float32)

        # Initialize object memory
        object_memory = self.coord_encoder.encode([[0, 0]])
        last_features = None

        # Get all event frame files
        data_files = natsorted([
            f for f in os.listdir(obj_path)
            if os.path.isfile(os.path.join(obj_path, f)) and f != '.DS_Store'
        ])

        print(f"  Processing {len(data_files)} event frames...")

        processed_count = 0
        for data_file in data_files:
            img_path = os.path.join(obj_path, data_file)

            # Process frame
            new_roi, image_features, coords = self.process_frame(
                img_path, saliency_map, resolution
            )

            if new_roi is not None:
                # Update memory
                object_memory = (self.config.GAMMA * object_memory +
                                 (1 - self.config.GAMMA) * new_roi)
                last_features = image_features
                processed_count += 1

            # Memory cleanup
            self._cleanup_memory()
            saliency_map.fill(0)

        print(f"  Successfully processed {processed_count}/{len(data_files)} frames")
        return object_memory, last_features

    def _cleanup_memory(self):
        """Clean up GPU memory"""
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

    def save_results(self, obj_name, object_memory, image_features, save_path, seq_label=None):
        """Save processing results. If seq_label is provided, save image_features under save_path/obj_name/seq_label/"""
        os.makedirs(save_path, exist_ok=True)

        # Save object memory (kept at top-level save_path for backward compatibility)
        memory_file = os.path.join(save_path, f'{obj_name}_memory.npy')
        np.save(memory_file, object_memory)

        # Save image features into obj_name/seq_label if seq_label provided, otherwise into obj_name
        if image_features is not None:
            if seq_label:
                feat_dir = os.path.join(save_path, obj_name, str(seq_label))
            else:
                feat_dir = os.path.join(save_path, obj_name)
            os.makedirs(feat_dir, exist_ok=True)

            # Name the features file using object and sequence for clarity when seq_label given
            if seq_label:
                features_file = os.path.join(feat_dir, f'{obj_name}_{seq_label}_image_features.npy')
            else:
                features_file = os.path.join(feat_dir, f'{obj_name}_image_features.npy')

            np.save(features_file, image_features)

        print(f"  ✅ Saved {obj_name} memory and features{f' (seq: {seq_label})' if seq_label else ''}")


def main():
    """Main processing pipeline"""
    # Validate paths
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model path does not exist: {MODEL_PATH}")
        return

    if not os.path.exists(PATH_DATA):
        print(f"❌ Data path does not exist: {PATH_DATA}")
        return

    # Initialize processor
    try:
        processor = EventFrameProcessor(MODEL_PATH)
    except Exception as e:
        print(f"❌ Failed to initialize processor: {e}")
        return

    # Get objects to process
    objects = natsorted([
        d for d in os.listdir(PATH_DATA)
        if os.path.isdir(os.path.join(PATH_DATA, d)) and not d.startswith('.')
    ])

    if not objects:
        print(f"❌ No objects found in {PATH_DATA}")
        return

    print(f"Found {len(objects)} objects to process: {objects}")

    # Process each object
    success_count = 0
    for obj in objects:
        # get the sequences for the object
        sequences = natsorted([
            d for d in os.listdir(os.path.join(PATH_DATA, obj))
            if os.path.isdir(os.path.join(PATH_DATA, obj, d)) and not d.startswith('.')
        ])

        for seq_label in sequences:
            obj_path = os.path.join(PATH_DATA, obj, seq_label)

            try:
                object_memory, image_features = processor.process_object(obj_path, obj)
                processor.save_results(obj, object_memory, image_features, 
                                       MEMORY_SAVE_PATH, seq_label=seq_label)
                success_count += 1

            except Exception as e:
                print(f"❌ Failed to process object {obj}: {e}")
                continue

    print(f"\n✅ Processing complete!")
    print(f"Successfully processed: {success_count}/{len(objects)} objects")
    print(f"Results saved to: {MEMORY_SAVE_PATH}")


if __name__ == '__main__':
    main()