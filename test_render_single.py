#!/usr/bin/env python3
"""Test rendering a single object"""

import mujoco
import numpy as np
import imageio
import os

# Test with dog object
xml_path = "CRIB Data/mujoco_objects/dog/dog.xml"
output_dir = "CRIB Data/test_render"
os.makedirs(output_dir, exist_ok=True)

print(f"Loading model from: {xml_path}")

try:
    # Load the model
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    print(f"Model loaded successfully!")
    print(f"  Bodies: {model.nbody}")
    print(f"  Geoms: {model.ngeom}")
    print(f"  Meshes: {model.nmesh}")
    print(f"  Cameras: {model.ncam}")

    # Create renderer
    renderer = mujoco.Renderer(model, 480, 640)

    # Use default camera (don't specify camera ID)
    print(f"Using default camera")

    # Render a few frames
    for frame_idx in range(4):
        angle = 2 * np.pi * frame_idx / 4

        # Reset simulation
        mujoco.mj_resetData(model, data)

        # Set rotation
        data.qpos[3] = np.cos(angle / 2)  # qw
        data.qpos[6] = np.sin(angle / 2)  # qz

        # Update
        mujoco.mj_forward(model, data)

        # Render with default camera
        renderer.update_scene(data)
        pixels = renderer.render()

        # Save
        frame_path = os.path.join(output_dir, f"test_frame_{frame_idx}.png")
        imageio.imwrite(frame_path, pixels)
        print(f"Saved frame {frame_idx}: {frame_path}")

    renderer.close()
    print("\n✓ Test rendering successful!")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
