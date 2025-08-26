import numpy as np
from attention_helpers import initialise_attention, run_attention
import torch
import os
from natsort import natsorted
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import cv2
import gc

# Set MPS memory management - Remove high watermark to avoid ratio errors
if 'PYTORCH_MPS_HIGH_WATERMARK_RATIO' in os.environ:
    del os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO']
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Disable upper limit

# Set matplotlib to use the 'TkAgg' backend for interactive plots
matplotlib.use('TkAgg')

transform = T.Compose([
    T.Grayscale(),
    T.ToTensor(),
])


ROOT = '/media/matt/bigdata/DATA/CRIB/'
SOURCE_PATH = ROOT + 'train_event_frames/'
BBOX_BASE_PATH = ROOT + 'bboxes/'
BATCH_SIZE = 64
WIDTH, HEIGHT = 400, 400
ROI_SIZE = 100  # Fixed size for bounding boxes
VISUALIZATION_FLAG = False


class Config:
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


def create_bbox_directory_for_object(bbox_base_path, obj_name):
    """Create bbox directory for a specific object class"""
    bbox_dir = os.path.join(bbox_base_path, obj_name)
    os.makedirs(bbox_dir, exist_ok=True)
    print(f"Created/verified directory: {bbox_dir}")
    return bbox_dir


def save_roi(roi_image, save_path, filename_base, img_format='png'):
    """Save ROI image to specified path"""
    try:
        filename = f"{filename_base}_roi.{img_format}"
        full_path = os.path.join(save_path, filename)

        # Ensure roi_image is uint8
        if roi_image.dtype != np.uint8:
            roi_image = roi_image.astype(np.uint8)

        # Convert to PIL Image
        roi_pil = Image.fromarray(roi_image, mode='L')  # 'L' for grayscale

        # Make sure directory exists
        os.makedirs(save_path, exist_ok=True)

        # Save the image
        roi_pil.save(full_path)

        return full_path

    except Exception as e:
        print(f"✗ ERROR saving ROI: {e}")
        return None


def cleanup_memory(device, force_gc=False):
    """Comprehensive memory cleanup with error handling"""
    try:
        # Clear GPU cache
        if device.type == "mps":
            torch.mps.empty_cache()
            # Try synchronize to clear any pending operations
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Force garbage collection
        if force_gc:
            gc.collect()

    except Exception as e:
        # Silently handle memory cleanup errors to prevent cascade failures
        pass


def visualize_roi_extraction_batch(source_path, device, config, visualisationFLAG, box_size, bbox_base_path,
                                   batch_files):
    """Process a batch of files with memory management"""

    # Initialize attention network
    net_attention = initialise_attention(device, config.ATTENTION_PARAMS)

    # Processing parameters - reduced for memory efficiency
    max_x, max_y = WIDTH, HEIGHT  # Reduced from 400x400
    resolution = (max_y, max_x)

    if visualisationFLAG:
        # Create visualization
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Keep track of processed objects to create directories on-demand
    processed_objects = set()

    # Process batch files
    for idx, (obj, img_path, data_file) in enumerate(batch_files):
        try:
            #print(f"Processing batch file {idx + 1}/{len(batch_files)}: {obj}/{data_file}")

            # Create bbox directory for this object if we haven't seen it before
            if obj not in processed_objects:
                bbox_save_path = create_bbox_directory_for_object(bbox_base_path, obj)
                processed_objects.add(obj)
            else:
                bbox_save_path = os.path.join(bbox_base_path, obj)

            # Load and preprocess image
            img = Image.open(img_path)
            window = transform(img)
            window_original = window.clone()

            # Initialize saliency map and coordinates
            saliency_map = np.zeros((max_y, max_x), dtype=np.float32)
            salmax_coords = np.zeros((2,), dtype=np.int32)

            # Run attention mechanism
            saliency_map[:], salmax_coords[:] = run_attention(
                window, net_attention, device, resolution, config.ATTENTION_PARAMS['num_pyr']
            )

            # Convert window to displayable format
            window_img = window.detach().cpu().numpy().squeeze(0)
            window_img = (window_img * 255).clip(0, 255).astype(np.uint8)

            # Extract coordinates
            y, x = salmax_coords[0], salmax_coords[1]

            # Calculate bounding box with fixed size
            x1 = max(x - (box_size // 2), 0)
            y1 = max(y - (box_size // 2), 0)
            x2 = x1 + box_size
            y2 = y1 + box_size

            # Ensure bbox does not exceed image boundaries
            x2 = min(x2, window_img.shape[1])
            y2 = min(y2, window_img.shape[0])
            x1 = x2 - box_size if x2 - x1 < box_size and x2 - box_size >= 0 else x1
            y1 = y2 - box_size if y2 - y1 < box_size and y2 - box_size >= 0 else y1

            # Extract ROI from original image
            roi_extracted = window_img[y1:y2, x1:x2]

            # Save ROI to corresponding bbox directory
            filename_base = os.path.splitext(data_file)[0]  # Remove extension
            saved_path = save_roi(roi_extracted, bbox_save_path, filename_base)

            if saved_path is None:
                print(f"✗ Failed to save ROI for {data_file}")
            #else:
            #    print(f"✓ ROI saved successfully for {data_file}")

            # Visualization (optional)
            if visualisationFLAG:
                # Create ROI visualization using mask approach
                mask = np.zeros_like(window_img)
                mask[y1:y2, x1:x2] = window_img[y1:y2, x1:x2]
                roi_visualization = mask

                # Clear previous plots
                ax1.clear()
                ax2.clear()
                ax3.clear()

                # Plot event frame with ROI box overlay
                ax1.imshow(window_img)
                ax1.set_title(f'Event Frame - {obj}')
                ax1.axis('off')
                # Add ROI rectangle overlay
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                         linewidth=2, edgecolor='red', facecolor='none')
                ax1.add_patch(rect)
                # Add center point
                ax1.plot(x, y, 'w+', markersize=12, markeredgewidth=3)

                # Plot saliency map with center point
                ax2.imshow(saliency_map, cmap='jet', vmin=0, vmax=255)
                ax2.set_title('Saliency Map')
                ax2.axis('off')
                ax2.plot(x, y, 'w+', markersize=12, markeredgewidth=3)

                # Plot ROI visualization
                ax3.imshow(roi_visualization)
                ax3.set_title('ROI on Salient Point')
                ax3.axis('off')

                plt.tight_layout()
                plt.draw()
                plt.pause(1.0)  # Reduced pause time

            # COMPREHENSIVE MEMORY CLEANUP WITH ERROR HANDLING
            try:
                # Delete all local variables
                del window, window_original, img
                del window_img, roi_extracted, saliency_map, salmax_coords

                if visualisationFLAG:
                    del mask, roi_visualization

                # Clear GPU cache with error handling
                cleanup_memory(device)

                # More aggressive cleanup every 25 images (reduced from 50)
                if idx % 25 == 0:
                    print(f"Aggressive memory cleanup at image {idx}")
                    cleanup_memory(device, force_gc=True)

            except Exception as cleanup_error:
                # Silent cleanup - don't print errors to avoid log spam
                pass

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            # Cleanup on error
            cleanup_memory(device)
            continue

    # Final cleanup for batch
    del net_attention
    cleanup_memory(device)

    if visualisationFLAG:
        plt.close(fig)


def get_all_files(source_path):
    """Get all files to process"""
    objects = natsorted([d for d in os.listdir(source_path)
                         if os.path.isdir(os.path.join(source_path, d))])

    all_files = []
    for obj in objects:
        obj_path = os.path.join(source_path, obj)
        data_files = natsorted([f for f in os.listdir(obj_path)
                                if f.lower().endswith(('.png', '.jpg', '.jpeg')) and f != '.DS_Store'])
        for data_file in data_files:
            all_files.append((obj, os.path.join(obj_path, data_file), data_file))

    return all_files


def main():
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = Config()

    # Get all files
    all_files = get_all_files(SOURCE_PATH)
    total_files = len(all_files)
    print(f"Total files to process: {total_files}")

    # PROCESS IN BATCHES TO AVOID MEMORY ISSUES

    for batch_start in range(0, total_files, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_files)
        batch_files = all_files[batch_start:batch_end]

        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (total_files - 1) // BATCH_SIZE + 1

        print(f"\n{'=' * 60}")
        print(f"Processing batch {batch_num}/{total_batches}")
        print(f"Files {batch_start + 1} to {batch_end} of {total_files}")
        print(f"{'=' * 60}")

        try:
            # Process this batch
            visualize_roi_extraction_batch(
                SOURCE_PATH, device, config, VISUALIZATION_FLAG,
                ROI_SIZE, BBOX_BASE_PATH, batch_files
            )

            # Aggressive cleanup between batches
            print(f"Completed batch {batch_num}, cleaning memory...")
            cleanup_memory(device, force_gc=True)

            print(f"✓ Batch {batch_num} completed successfully")

        except Exception as e:
            print(f"✗ Error in batch {batch_num}: {e}")
            # More aggressive recovery
            cleanup_memory(device, force_gc=True)
            # Add a small delay to let system recover
            import time
            time.sleep(1)
            print(f"Attempting to continue with next batch...")
            continue

    print(f"\n{'=' * 60}")
    print("✓ All batches completed!")
    print(f"✓ Processed {total_files} files total")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
