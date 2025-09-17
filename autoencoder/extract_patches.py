#!/usr/bin/env python3
"""
Script to extract random patches from CRIB dataset images and split into train/test sets.
Usage: python extract_patches.py --M <num_patches> --N <patch_size> [--crop <pixels>] [--test_split <ratio>]
"""

import os
import argparse
import random
from pathlib import Path
import numpy as np
from PIL import Image
import shutil
from sklearn.model_selection import train_test_split

def load_and_extract_patches(image_path, M, N, crop=0):
    """Extract M random patches of size NxN from an image with optional cropping."""
    try:
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        img_array = np.array(image)
        height, width = img_array.shape[:2]

        # Apply cropping if specified
        if crop > 0:
            # Crop from all edges
            crop_top = crop
            crop_bottom = height - crop
            crop_left = crop
            crop_right = width - crop

            if crop_bottom <= crop_top or crop_right <= crop_left:
                print(f"Warning: Crop size {crop} is too large for image {image_path} ({width}x{height})")
                return []

            img_array = img_array[crop_top:crop_bottom, crop_left:crop_right]
            height, width = img_array.shape[:2]

        if height < N or width < N:
            print(f"Warning: Image {image_path} (after cropping) is smaller than patch size {N}x{N}")
            return []

        patches = []
        for _ in range(M):
            # Random top-left corner for patch
            top = random.randint(0, height - N)
            left = random.randint(0, width - N)

            # Extract patch
            patch = img_array[top:top+N, left:left+N]
            patches.append(patch)

        return patches

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return []

def save_patch(patch, save_path):
    """Save a patch as PNG image."""
    patch_img = Image.fromarray(patch.astype(np.uint8))
    patch_img.save(save_path)

def main():
    parser = argparse.ArgumentParser(description='Extract random patches from CRIB dataset')
    parser.add_argument('--M', type=int, required=True, help='Number of patches to extract per image')
    parser.add_argument('--N', type=int, required=True, help='Patch size (NxN)')
    parser.add_argument('--crop', type=int, default=0, help='Crop pixels from each edge before extracting patches (default: 0)')
    parser.add_argument('--test_split', type=float, default=0.2, help='Test split ratio (default: 0.2)')
    parser.add_argument('--data_path', type=str, default='/media/matt/bigdata/DATA/CRIB/train_event_frames',
                       help='Path to CRIB dataset')
    parser.add_argument('--output_dir', type=str, default='./patches_data', help='Output directory')

    args = parser.parse_args()

    # Create output directories
    output_path = Path(args.output_dir)
    train_dir = output_path / 'train'
    test_dir = output_path / 'test'

    # Remove existing directories if they exist
    if output_path.exists():
        shutil.rmtree(output_path)

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {args.M} patches of size {args.N}x{args.N} from each image...")
    if args.crop > 0:
        print(f"Cropping {args.crop} pixels from each edge before patch extraction")
    print(f"Output directory: {output_path}")
    print(f"Train/test split: {1-args.test_split:.1f}/{args.test_split:.1f}")

    # Collect all image paths
    data_path = Path(args.data_path)
    all_patches = []
    patch_counter = 0

    # Iterate through objects
    for obj_dir in data_path.iterdir():
        if not obj_dir.is_dir():
            continue

        print(f"Processing object: {obj_dir.name}")

        # Iterate through sequences (0-5)
        for seq_dir in obj_dir.iterdir():
            if not seq_dir.is_dir():
                continue

            print(f"  Processing sequence: {seq_dir.name}")

            # Process all images in sequence
            for img_path in sorted(seq_dir.glob('*.png')):
                patches = load_and_extract_patches(img_path, args.M, args.N, args.crop)

                for patch in patches:
                    patch_info = {
                        'patch': patch,
                        'object': obj_dir.name,
                        'sequence': seq_dir.name,
                        'original_file': img_path.name,
                        'patch_id': patch_counter
                    }
                    all_patches.append(patch_info)
                    patch_counter += 1

    print(f"Total patches extracted: {len(all_patches)}")

    # Split into train/test
    if len(all_patches) == 0:
        print("No patches extracted!")
        return

    train_patches, test_patches = train_test_split(
        all_patches,
        test_size=args.test_split,
        random_state=42,
        stratify=[p['object'] for p in all_patches]  # Stratify by object to ensure balanced split
    )

    print(f"Train patches: {len(train_patches)}")
    print(f"Test patches: {len(test_patches)}")

    # Save train patches
    print("Saving train patches...")
    for i, patch_info in enumerate(train_patches):
        filename = f"{patch_info['object']}_{patch_info['sequence']}_{patch_info['original_file'][:-4]}_{patch_info['patch_id']:06d}.png"
        save_path = train_dir / filename
        save_patch(patch_info['patch'], save_path)

    # Save test patches
    print("Saving test patches...")
    for i, patch_info in enumerate(test_patches):
        filename = f"{patch_info['object']}_{patch_info['sequence']}_{patch_info['original_file'][:-4]}_{patch_info['patch_id']:06d}.png"
        save_path = test_dir / filename
        save_patch(patch_info['patch'], save_path)

    print("Done!")
    print(f"Train patches saved to: {train_dir}")
    print(f"Test patches saved to: {test_dir}")

if __name__ == "__main__":
    main()