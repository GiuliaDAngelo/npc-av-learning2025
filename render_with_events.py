#!/usr/bin/env python3
"""
MuJoCo rendering with real-time event conversion using IEBCS.
Displays original rendered view and event frames side-by-side.
Optional saccade detection using attention mechanism.
"""

import mujoco
import numpy as np
import os
import cv2
import sys
import argparse
import torch
import torchvision.transforms as T
from PIL import Image

# Add IEBCS to path
sys.path.append("IEBCS/src")
from dvs_sensor import DvsSensor
from event_buffer import EventBuffer

# Import attention mechanism
from attention_helpers import initialise_attention, run_attention

# Configuration
OBJECTS_DIR = "CRIB Data/mujoco_objects"
WIDTH = 640
HEIGHT = 480
CAMERA_NAME = None  # Use default camera

# Rotation control parameters
INITIAL_YAW_SPEED = 0.05   # Initial rotation speed around Z-axis (radians per frame)
INITIAL_PITCH_SPEED = 0.0  # Initial rotation speed around Y-axis (radians per frame)
SPEED_INCREMENT = 0.01     # How much speed changes with arrow keys

# DVS Sensor parameters (from IEBCS example)
TH_POS = 0.4        # ON threshold = 50% (ln(1.5) = 0.4)
TH_NEG = 0.4        # OFF threshold = 50%
TH_NOISE = 0.01     # standard deviation of threshold noise
LAT = 100           # latency in us
TAU = 40            # front-end time constant at 1 klux in us
JIT = 10            # temporal jitter standard deviation in us
BGNP = 0.1          # ON event noise rate in events / pixel / s
BGNN = 0.01         # OFF event noise rate in events / pixel / s
REF = 100           # refractory period in us
DT = 33333          # time between frames in us (30 fps)

# Attention mechanism parameters (from Code0GenerateBboxes)
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
ROI_SIZE = 100  # Size of saccade bounding box


def process_saccade_vsa(image_patch, saccade_center, rotation_state, dino_model, dino_transform, dino_device):
    """
    VSA processing stub for saccade-based learning.

    Args:
        image_patch: numpy array of the image patch around saccade (ROI_SIZE x ROI_SIZE x 3)
        saccade_center: tuple of (x, y) coordinates of saccade center
        rotation_state: dict containing rotation information {'yaw': float, 'pitch': float, 'quaternion': tuple}
        dino_model: DINO model for extracting embeddings
        dino_transform: Transform for preprocessing image for DINO
        dino_device: torch device for DINO
    """
    # Convert BGR to RGB for DINO
    image_patch_rgb = cv2.cvtColor(image_patch, cv2.COLOR_BGR2RGB)

    # Convert to PIL Image
    pil_image = Image.fromarray(image_patch_rgb)

    # Preprocess for DINO
    input_tensor = dino_transform(pil_image).unsqueeze(0).to(dino_device)

    # Extract DINO embeddings
    with torch.no_grad():
        embeddings = dino_model(input_tensor)

    print(f"\n=== VSA Saccade Processing ===")
    print(f"Saccade center: ({saccade_center[0]}, {saccade_center[1]})")
    print(f"Image patch shape: {image_patch.shape}")
    print(f"DINO embedding shape: {embeddings.shape}, {embeddings.min():.3f} to {embeddings.max():.3f}")
    print(f"Rotation - Yaw: {rotation_state['yaw']:.3f} rad, Pitch: {rotation_state['pitch']:.3f} rad")
    print(f"Quaternion: [{rotation_state['quaternion'][0]:.3f}, {rotation_state['quaternion'][1]:.3f}, "
          f"{rotation_state['quaternion'][2]:.3f}, {rotation_state['quaternion'][3]:.3f}]")
    print("=" * 40)

    return embeddings


class EventFrameRenderer:
    """Renders time surface from events"""

    def __init__(self, width, height, tau=40000):
        self.width = width
        self.height = height
        self.tau = tau  # decay constant in us
        self.time = 0
        self.time_surface = np.zeros((height, width), dtype=np.uint64)
        self.pol_surface = np.zeros((height, width), dtype=np.uint8)

    def update(self, events, dt):
        """Update time surface with new events and return rendered frame"""
        # Update time surfaces
        if events.i > 0:
            self.time_surface[events.y[:events.i], events.x[:events.i]] = events.ts[:events.i]
            self.pol_surface[events.y[:events.i], events.x[:events.i]] = events.p[:events.i]

        self.time += dt

        # Render time surface with exponential decay
        img = np.ones((self.height, self.width), dtype=np.float32) * 125

        # Find pixels with recent events
        ind = np.where(self.time_surface > 0)
        if len(ind[0]) > 0:
            # Calculate decay based on time since event
            decay = np.exp(-(self.time - self.time_surface[ind].astype(np.float32)) / self.tau)
            # Polarity: 1 for ON (positive), 0 for OFF (negative)
            # Map to: ON events = bright (positive), OFF events = dark (negative)
            polarity_value = self.pol_surface[ind] * 2.0 - 1.0  # Maps 1->1, 0->-1
            img[ind] = 125 + polarity_value * decay * 125

        # Convert to uint8 and apply colormap
        img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
        img_color = cv2.applyColorMap(img_uint8, cv2.COLORMAP_VIRIDIS)

        return img_color


def render_rotating_object_with_events(xml_path, obj_name, enable_saccades=False):
    """
    Render object with real-time event conversion.
    Shows both original view and event frame side-by-side.

    Args:
        xml_path: Path to MJCF file
        obj_name: Name of object
        enable_saccades: If True, run attention mechanism and show saccade location

    Returns:
        True if completed normally, False if user quit
    """
    try:
        # Load MuJoCo model
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)

        # Create MuJoCo renderer
        renderer = mujoco.Renderer(model, HEIGHT, WIDTH)

        # Set up camera
        if CAMERA_NAME is None:
            camera_id = -1
        else:
            try:
                camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
            except:
                print(f"  Warning: Camera '{CAMERA_NAME}' not found, using default")
                camera_id = -1

        # Initialize DVS sensor
        print(f"  Initializing DVS sensor ({WIDTH}x{HEIGHT})...")
        dvs = DvsSensor("RealTimeDVS")
        dvs.initCamera(WIDTH, HEIGHT,
                      lat=LAT, jit=JIT, ref=REF, tau=TAU,
                      th_pos=TH_POS, th_neg=TH_NEG, th_noise=TH_NOISE,
                      bgnp=BGNP, bgnn=BGNN)

        # Render first frame to initialize DVS
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        if camera_id >= 0:
            renderer.update_scene(data, camera=camera_id)
        else:
            renderer.update_scene(data)
        first_frame = renderer.render()

        # Convert first frame to luminance and initialize DVS
        first_frame_rgb = first_frame  # Already RGB from MuJoCo
        first_luv = cv2.cvtColor(first_frame_rgb, cv2.COLOR_RGB2LUV)
        first_lum = first_luv[:, :, 0] / 255.0 * 1e4  # Scale to 10 klux
        dvs.init_image(first_lum)

        # Initialize event frame renderer
        event_renderer = EventFrameRenderer(WIDTH, HEIGHT, tau=3*DT)

        # Initialize attention network and DINO if saccades enabled
        net_attention = None
        dino_model = None
        dino_transform = None
        dino_device = None
        device = None
        transform = None
        if enable_saccades:
            print(f"  Initializing attention network for saccades...")
            device = torch.device("mps" if torch.backends.mps.is_available()
                                else "cuda" if torch.cuda.is_available() else "cpu")
            print(f"    Using device for attention: {device}")
            net_attention = initialise_attention(device, ATTENTION_PARAMS)
            transform = T.Compose([
                T.ToTensor(),
            ])

            # Initialize DINO model on CPU (MPS doesn't support bicubic interpolation)
            print(f"  Initializing DINO model...")
            dino_device = torch.device("cpu")
            print(f"    Using device for DINO: {dino_device} (MPS doesn't support bicubic interpolation)")
            dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            dino_model = dino_model.to(dino_device)
            dino_model.eval()

            # DINO preprocessing transform
            dino_transform = T.Compose([
                T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            print(f"    DINO model loaded")

        # Create display window
        window_name = f"Real-time Events: {obj_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, WIDTH * 2, HEIGHT)

        print(f"  Rendering {obj_name}...")
        if enable_saccades:
            print(f"  Saccades enabled - showing attention on event frames")
        print(f"  Controls:")
        print(f"    Left/Right arrows: Adjust yaw rotation speed")
        print(f"    Up/Down arrows: Adjust pitch rotation speed")
        print(f"    'q' or ESC: Exit")

        # Initialize rotation state
        yaw_angle = 0.0      # Rotation around Z-axis
        pitch_angle = 0.0    # Rotation around Y-axis
        yaw_speed = INITIAL_YAW_SPEED
        pitch_speed = INITIAL_PITCH_SPEED
        frame_count = 0

        # Continuous rendering loop
        while True:
            # Update angles based on speeds
            yaw_angle += yaw_speed
            pitch_angle += pitch_speed

            # Reset simulation
            mujoco.mj_resetData(model, data)

            # Compute quaternion from yaw and pitch (ZYX Euler angles)
            # Quaternion for yaw (Z-axis rotation)
            qz_w = np.cos(yaw_angle / 2)
            qz_x = 0
            qz_y = 0
            qz_z = np.sin(yaw_angle / 2)

            # Quaternion for pitch (Y-axis rotation)
            qy_w = np.cos(pitch_angle / 2)
            qy_x = 0
            qy_y = np.sin(pitch_angle / 2)
            qy_z = 0

            # Combine quaternions: q_total = qz * qy
            data.qpos[3] = qz_w * qy_w - qz_z * qy_y  # qw
            data.qpos[4] = qz_w * qy_x + qz_z * qy_y  # qx
            data.qpos[5] = qz_w * qy_y + qz_z * qy_w  # qy
            data.qpos[6] = qz_w * qy_z + qz_z * qy_w  # qz

            # Step the simulation
            mujoco.mj_forward(model, data)

            # Render the scene
            if camera_id >= 0:
                renderer.update_scene(data, camera=camera_id)
            else:
                renderer.update_scene(data)
            pixels = renderer.render()

            # Convert to luminance for DVS
            pixels_luv = cv2.cvtColor(pixels, cv2.COLOR_RGB2LUV)
            luminance = pixels_luv[:, :, 0] / 255.0 * 1e4

            # Generate events
            events = dvs.update(luminance, DT)

            # Render event frame
            event_frame = event_renderer.update(events, DT)

            # Run attention mechanism on event frame if enabled
            saccade_x, saccade_y = None, None
            if enable_saccades and events.i > 0:
                # Convert event frame to grayscale for attention
                event_gray = cv2.cvtColor(event_frame, cv2.COLOR_BGR2GRAY)

                # Convert to tensor (don't add batch dimension - run_attention creates its own batching)
                event_tensor = transform(event_gray)

                # Run attention
                saliency_map, salmax_coords = run_attention(
                    event_tensor, net_attention, device,
                    (HEIGHT, WIDTH), ATTENTION_PARAMS['num_pyr']
                )

                # Extract coordinates
                saccade_y, saccade_x = salmax_coords[0], salmax_coords[1]

            # Convert original frame to BGR for display
            display_original = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)

            # Create copy of event frame for overlay
            event_frame_display = event_frame.copy()

            # Overlay saccade location on both frames if enabled
            if enable_saccades and saccade_x is not None:
                # Calculate bounding box
                x1 = max(saccade_x - (ROI_SIZE // 2), 0)
                y1 = max(saccade_y - (ROI_SIZE // 2), 0)
                x2 = min(x1 + ROI_SIZE, WIDTH)
                y2 = min(y1 + ROI_SIZE, HEIGHT)

                # Adjust if box extends beyond boundaries
                if x2 - x1 < ROI_SIZE and x1 > 0:
                    x1 = max(x2 - ROI_SIZE, 0)
                if y2 - y1 < ROI_SIZE and y1 > 0:
                    y1 = max(y2 - ROI_SIZE, 0)

                # Extract image patch from event frame
                image_patch = event_frame[y1:y2, x1:x2]

                # Prepare rotation state
                rotation_state = {
                    'yaw': yaw_angle,
                    'pitch': pitch_angle,
                    'quaternion': (data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6])
                }

                # Process with VSA (stub function)
                process_saccade_vsa(image_patch, (saccade_x, saccade_y), rotation_state,
                                   dino_model, dino_transform, dino_device)

                # Draw on EVENT frame
                # Draw crosshair at saccade location
                cv2.drawMarker(event_frame_display, (saccade_x, saccade_y),
                             (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
                # Draw bounding box
                cv2.rectangle(event_frame_display, (x1, y1), (x2, y2),
                            (0, 255, 0), 2)

                # Draw on ORIGINAL frame
                # Draw crosshair at saccade location
                cv2.drawMarker(display_original, (saccade_x, saccade_y),
                             (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
                # Draw bounding box
                cv2.rectangle(display_original, (x1, y1), (x2, y2),
                            (0, 255, 0), 2)

            # Combine original and event frame side-by-side
            combined = np.hstack([display_original, event_frame_display])

            # Add labels
            cv2.putText(combined, "Original", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            label = "Events + Saccades" if enable_saccades else "Events"
            cv2.putText(combined, label, (WIDTH + 10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Show rotation speeds
            speed_info = f"Yaw: {yaw_speed:.3f} | Pitch: {pitch_speed:.3f} rad/frame"
            cv2.putText(combined, speed_info, (10, HEIGHT - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Show event info
            event_info = f"{events.i} events"
            if enable_saccades and saccade_x is not None:
                event_info += f" | Saccade: ({saccade_x},{saccade_y})"
            cv2.putText(combined, event_info, (WIDTH + 10, HEIGHT - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Display
            cv2.imshow(window_name, combined)

            # Check for keyboard input
            key = cv2.waitKey(30)
            if key == 27 or key == ord('q') or key == ord('Q'):
                print(f"  Stopped by user")
                cv2.destroyWindow(window_name)
                renderer.close()
                return False
            elif key != -1:
                # macOS arrow key codes: Left=2, Right=3, Up=0, Down=1
                if key == 2 or key == 81:  # Left arrow
                    yaw_speed -= SPEED_INCREMENT
                    print(f"  Yaw speed: {yaw_speed:.3f} rad/frame")
                elif key == 3 or key == 83:  # Right arrow
                    yaw_speed += SPEED_INCREMENT
                    print(f"  Yaw speed: {yaw_speed:.3f} rad/frame")
                elif key == 0 or key == 82:  # Up arrow
                    pitch_speed += SPEED_INCREMENT
                    print(f"  Pitch speed: {pitch_speed:.3f} rad/frame")
                elif key == 1 or key == 84:  # Down arrow
                    pitch_speed -= SPEED_INCREMENT
                    print(f"  Pitch speed: {pitch_speed:.3f} rad/frame")

            frame_count += 1

        # Close display window and renderer
        cv2.destroyWindow(window_name)
        renderer.close()

        return True

    except Exception as e:
        print(f"  ✗ Error rendering {obj_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Render objects with real-time event conversion."""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Render rotating 3D objects with real-time event conversion')
    parser.add_argument('--object', type=str, default=None,
                        help='Render only a specific object by name (default: dog)')
    parser.add_argument('--saccades', action='store_true',
                        help='Enable saccade detection using attention mechanism')
    args = parser.parse_args()

    # Default to dog if no object specified
    obj_name = args.object if args.object else "dog"

    # Find XML file
    obj_dir = os.path.join(OBJECTS_DIR, obj_name)
    xml_path = os.path.join(obj_dir, f"{obj_name}.xml")

    if not os.path.exists(xml_path):
        print(f"Error: Object '{obj_name}' not found at {xml_path}")
        return

    print(f"Rendering object: {obj_name}")
    print(f"Settings: {WIDTH}x{HEIGHT}, continuous rotation")
    print(f"DVS parameters: th_pos={TH_POS}, th_neg={TH_NEG}, tau={TAU}us")
    if args.saccades:
        print(f"Saccades: ENABLED (ROI size: {ROI_SIZE}x{ROI_SIZE})")
    print("=" * 80)

    # Render the object
    result = render_rotating_object_with_events(xml_path, obj_name, enable_saccades=args.saccades)

    cv2.destroyAllWindows()

    if result:
        print("\n✓ Rendering complete!")
    else:
        print("\n⊘ Rendering stopped")


if __name__ == "__main__":
    main()
