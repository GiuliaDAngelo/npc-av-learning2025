#!/usr/bin/env python3
"""
MuJoCo rendering script for rotating 3D objects.
Loads objects from MJCF files, rotates them, and saves rendered frames.
"""

import mujoco
import numpy as np
import os
import imageio
from pathlib import Path
import argparse
import cv2

# Configuration
OBJECTS_DIR = "CRIB Data/mujoco_objects"
OUTPUT_DIR = "CRIB Data/rendered_objects"
WIDTH = 640
HEIGHT = 480
NUM_FRAMES = 36  # Number of frames per rotation (10 degrees per frame)
CAMERA_NAME = None  # Which camera to use: "fixed", "top", "side", or None for default


def render_rotating_object(xml_path, output_dir, obj_name, display=False, save=True):
    """
    Load an object from MJCF XML, rotate it, and optionally display/save rendered frames.

    Args:
        xml_path: Path to the MJCF XML file
        output_dir: Directory to save rendered frames
        obj_name: Name of the object
        display: If True, display frames in a window
        save: If True, save frames to disk

    Returns:
        True if completed normally, False if user quit
    """
    try:
        # Load the model
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)

        # Create renderer
        renderer = mujoco.Renderer(model, HEIGHT, WIDTH)

        # Find the camera (use -1 for default camera if CAMERA_NAME is None)
        if CAMERA_NAME is None:
            camera_id = -1
        else:
            try:
                camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
            except:
                print(f"  Warning: Camera '{CAMERA_NAME}' not found, using default")
                camera_id = -1

        # Create output directory for this object if saving
        if save:
            obj_output_dir = os.path.join(output_dir, obj_name)
            os.makedirs(obj_output_dir, exist_ok=True)

        frames = []
        window_name = f"Rendering: {obj_name}"

        if display:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, WIDTH, HEIGHT)

        # Render frames at different rotation angles
        for frame_idx in range(NUM_FRAMES):
            # Calculate rotation angle (in radians)
            angle = 2 * np.pi * frame_idx / NUM_FRAMES

            # Reset simulation
            mujoco.mj_resetData(model, data)

            # Set the object's rotation around Z-axis
            # The freejoint has 7 DOFs: [x, y, z, qw, qx, qy, qz]
            # We'll rotate around the Z-axis
            data.qpos[3] = np.cos(angle / 2)  # qw
            data.qpos[4] = 0  # qx
            data.qpos[5] = 0  # qy
            data.qpos[6] = np.sin(angle / 2)  # qz

            # Step the simulation to update positions
            mujoco.mj_forward(model, data)

            # Render the scene
            if camera_id >= 0:
                renderer.update_scene(data, camera=camera_id)
            else:
                renderer.update_scene(data)  # Use default camera
            pixels = renderer.render()

            # Display frame if requested
            if display:
                # Convert RGB to BGR for OpenCV
                display_frame = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
                cv2.imshow(window_name, display_frame)
                # Wait for a short time (30ms = ~30fps), allow ESC or 'q' to quit
                key = cv2.waitKey(30)
                if key == 27 or key == ord('q') or key == ord('Q'):  # ESC or 'q' key
                    print(f"  Stopped by user")
                    # Close display window and renderer
                    if display:
                        cv2.destroyWindow(window_name)
                    renderer.close()
                    return False  # Signal user quit

            # Save frame if requested
            if save:
                frame_path = os.path.join(obj_output_dir, f"frame_{frame_idx:03d}.png")
                imageio.imwrite(frame_path, pixels)
                frames.append(pixels)

            if frame_idx % 10 == 0:
                print(f"    Frame {frame_idx}/{NUM_FRAMES}")

        # Close display window if it was opened
        if display:
            cv2.destroyWindow(window_name)

        # Create video if frames were saved
        if save and frames:
            video_path = os.path.join(obj_output_dir, f"{obj_name}_rotation.mp4")
            imageio.mimsave(video_path, frames, fps=10)
            print(f"  ✓ Rendered {NUM_FRAMES} frames")
            print(f"  ✓ Saved video: {video_path}")
        elif display:
            print(f"  ✓ Displayed {NUM_FRAMES} frames")

        # Close renderer
        renderer.close()

        return True

    except Exception as e:
        print(f"  ✗ Error rendering {obj_name}: {e}")
        return False


def main():
    """Render all objects in the objects directory."""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Render rotating 3D objects with MuJoCo')
    parser.add_argument('--display', action='store_true',
                        help='Display rendering in a window')
    parser.add_argument('--nosave', action='store_true',
                        help='Do not save frames to disk (only display)')
    parser.add_argument('--object', type=str, default=None,
                        help='Render only a specific object by name')
    args = parser.parse_args()

    # Determine display and save modes
    display = args.display
    save = not args.nosave

    if not display and not save:
        print("Error: Must enable at least one of --display or saving (remove --nosave)")
        return

    # Create output directory if saving
    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find all XML files
    xml_files = []
    for obj_name in sorted(os.listdir(OBJECTS_DIR)):
        obj_dir = os.path.join(OBJECTS_DIR, obj_name)
        if os.path.isdir(obj_dir):
            # Filter by specific object if requested
            if args.object and obj_name != args.object:
                continue
            xml_path = os.path.join(obj_dir, f"{obj_name}.xml")
            if os.path.exists(xml_path):
                xml_files.append((obj_name, xml_path))

    if len(xml_files) == 0:
        if args.object:
            print(f"Error: Object '{args.object}' not found")
        else:
            print("Error: No objects found")
        return

    print(f"Found {len(xml_files)} object(s) to render")
    if save:
        print(f"Output directory: {OUTPUT_DIR}")
    print(f"Rendering settings: {WIDTH}x{HEIGHT}, {NUM_FRAMES} frames, camera: {CAMERA_NAME}")
    print(f"Display: {'Yes' if display else 'No'}, Save: {'Yes' if save else 'No'}")
    if display:
        print("Press 'q' or ESC to exit")
    print("=" * 80)

    success_count = 0
    fail_count = 0

    # Render each object
    user_quit = False
    for idx, (obj_name, xml_path) in enumerate(xml_files):
        print(f"\n[{idx+1}/{len(xml_files)}] Rendering: {obj_name}")

        result = render_rotating_object(xml_path, OUTPUT_DIR, obj_name, display=display, save=save)
        if result is False:
            # User pressed quit
            user_quit = True
            print("\nExiting due to user request...")
            break
        elif result:
            success_count += 1
        else:
            fail_count += 1

    # Clean up OpenCV windows
    if display:
        cv2.destroyAllWindows()

    print("\n" + "=" * 80)
    print(f"Rendering complete!")
    print(f"  Success: {success_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Total: {len(xml_files)}")
    if save:
        print(f"\nRendered frames saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
