#!/usr/bin/env python3
"""
MuJoCo rendering with real-time event conversion using IEBCS.
Includes a Deep Q-Learning agent to actively control object rotation and saccades
for a discrimination task.
"""

import mujoco
import numpy as np
import os
import cv2
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import sspspace
import random
from collections import deque
import math
import json
from datetime import datetime

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
INITIAL_YAW_SPEED = 0.05
INITIAL_PITCH_SPEED = 0.0
SPEED_INCREMENT = 0.01

# DVS Sensor parameters
TH_POS = 0.4
TH_NEG = 0.4
TH_NOISE = 0.01
LAT = 100
TAU = 40
JIT = 10
BGNP = 0.1
BGNN = 0.01
REF = 100
DT = 33333

# Attention mechanism parameters
ATTENTION_PARAMS = {
    'size_krn': 16, 'r0': 14, 'rho': 0.05, 'theta': np.pi * 3 / 2,
    'thetas': np.arange(0, 2 * np.pi, np.pi / 4), 'thick': 3,
    'fltr_resize_perc': [2, 2], 'offsetpxs': 0, 'offset': (0, 0),
    'num_pyr': 6, 'tau_mem': 0.3, 'stride': 1, 'out_ch': 1
}
ROI_SIZE = 100
GAMMA_VSA = 0.99  # VSA Memory update factor

# --- RL: New constants for the Deep Q-Learning Agent ---
K_SALIENCY = 5              # Agent can choose from top K salient points
NUM_MOVE_ACTIONS = 5        # inc/dec yaw, inc/dec pitch, do_nothing
NUM_SACCADE_ACTIONS = K_SALIENCY
TOTAL_ACTIONS = NUM_MOVE_ACTIONS + NUM_SACCADE_ACTIONS
SSP_DIM = 1000              # Must match WorkingMemory ssp_dim

# RL Hyperparameters
REPLAY_BUFFER_SIZE = 5000
BATCH_SIZE = 8
GAMMA_RL = 0.99             # RL discount factor
EPSILON_START = 0.9
EPSILON_END = 0.05
EPSILON_DECAY = 1000
TARGET_UPDATE_FREQ = 10     # Update target network every 10 episodes/steps

# --- RL: Helper function to get top K saliency points ---
def get_top_k_saliency_coords(saliency_map, k):
    """Finds the coordinates of the top K values in a saliency map."""
    # Flatten the map to find the top K values' indices
    flat_map = saliency_map.flatten()
    # Use argpartition for efficiency; it finds the K-th largest element's index
    # and ensures all elements after it are larger.
    # We negate the array to find the largest values.
    try:
        top_k_indices = np.argpartition(-flat_map, k)[:k]
    except ValueError: # Handle cases where map has fewer than K non-zero elements
        top_k_indices = np.argsort(-flat_map)[:k]

    # Convert the flat indices back to 2D coordinates
    coords = np.unravel_index(top_k_indices, saliency_map.shape)
    # Return as a list of (y, x) tuples
    return list(zip(coords[0], coords[1]))


# --- RL: Deep Q-Network Agent Implementation ---
class DiscriminationAgent:
    def __init__(self, state_dim, num_actions):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        print(f"  RL Agent using device: {self.device}")

        # Q-Network and Target Network for stability
        self.policy_net = self._create_network().to(self.device)
        self.target_net = self._create_network().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is only for inference

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)
        self.steps_done = 0

    def _create_network(self):
        """Creates the neural network architecture."""
        return nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions)
        )

    def choose_action(self, state):
        """Chooses an action using an epsilon-greedy policy."""
        # Epsilon decay
        epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                  math.exp(-1. * self.steps_done / EPSILON_DECAY)
        self.steps_done += 1

        if random.random() > epsilon:
            # Exploit: choose the best action from the policy network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return torch.argmax(q_values).item()
        else:
            # Explore: choose a random action
            return random.randrange(self.num_actions)

    def remember(self, state, action, reward, next_state):
        """Stores an experience tuple in the replay buffer."""
        self.replay_buffer.append((state, action, reward, next_state))

    def replay(self):
        """Trains the policy network using a batch of experiences from the buffer."""
        if len(self.replay_buffer) < BATCH_SIZE:
            return  # Not enough experiences to train

        # Sample a random batch
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        states, actions, rewards, next_states = zip(*minibatch)

        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)

        # 1. Get Q-values for current states: Q(s, a)
        current_q_values = self.policy_net(states).gather(1, actions)

        # 2. Get expected Q-values from next states using the target network
        # V(s') = max_a'(Q_target(s', a'))
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
        
        # Bellman equation: E[Q] = r + gamma * V(s')
        expected_q_values = rewards + (GAMMA_RL * next_q_values)

        # 3. Compute loss (Smooth L1 Loss is often more stable than MSE)
        loss = F.smooth_l1_loss(current_q_values, expected_q_values)

        # 4. Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_net(self):
        """Copies weights from the policy network to the target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        """Saves the agent's policy network weights."""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done
        }, path)
        print(f"  Agent saved to {path}")

    def load(self, path):
        """Loads the agent's policy network weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint['steps_done']
        print(f"  Agent loaded from {path}")

    def choose_action_greedy(self, state):
        """Chooses the best action without exploration (for inference)."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return torch.argmax(q_values).item()


class WorkingMemory:
    """VSA-based working memory for saccade-based object learning."""
    def __init__(self, dino_model, dino_transform, dino_device, ssp_dim=SSP_DIM):
        self.dino_model = dino_model
        self.dino_transform = dino_transform
        self.dino_device = dino_device
        self.ssp_dim = ssp_dim
        self.coord_encoder = sspspace.RandomSSPSpace(domain_dim=2, ssp_dim=ssp_dim)
        self.quat_encoder = sspspace.RandomSSPSpace(domain_dim=4, ssp_dim=ssp_dim)
        self.memory = np.zeros((1, ssp_dim))
        self.saccade_count = 0
        print(f"  Working Memory initialized (SSP dim: {ssp_dim})")

    def bind(self, a, b):
        a = np.atleast_2d(a)
        b = np.atleast_2d(b)
        return np.fft.ifft(np.fft.fft(a, axis=1) * np.fft.fft(b, axis=1), axis=1).real

    def process_saccade(self, image_patch, saccade_center, rotation_state):
        image_patch_rgb = cv2.cvtColor(image_patch, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_patch_rgb)
        input_tensor = self.dino_transform(pil_image).unsqueeze(0).to(self.dino_device)
        with torch.no_grad():
            dino_embedding = self.dino_model(input_tensor).cpu().numpy()
        if dino_embedding.shape[1] < self.ssp_dim:
            padding = np.zeros((1, self.ssp_dim - dino_embedding.shape[1]))
            dino_embedding = np.hstack([dino_embedding, padding])
        else:
            dino_embedding = dino_embedding[:, :self.ssp_dim]
        x, y = saccade_center
        coord_ssp = self.coord_encoder.encode([[x, y]])
        quat = rotation_state['quaternion']
        quat_ssp = self.quat_encoder.encode([[quat[0], quat[1], quat[2], quat[3]]])
        bound_img_coord = self.bind(dino_embedding, coord_ssp)
        bound_representation = self.bind(bound_img_coord, quat_ssp)
        self.memory = GAMMA_VSA * self.memory + (1 - GAMMA_VSA) * bound_representation
        self.saccade_count += 1
        return self.memory

    def get_memory(self):
        return self.memory.flatten()

    def reset_memory(self):
        self.memory = np.zeros((1, self.ssp_dim))
        self.saccade_count = 0
        print("  Working memory reset")


class EventFrameRenderer:
    def __init__(self, width, height, tau=40000):
        self.width, self.height, self.tau = width, height, tau
        self.time = 0
        self.time_surface = np.zeros((height, width), dtype=np.uint64)
        self.pol_surface = np.zeros((height, width), dtype=np.uint8)

    def update(self, events, dt):
        if events.i > 0:
            self.time_surface[events.y[:events.i], events.x[:events.i]] = events.ts[:events.i]
            self.pol_surface[events.y[:events.i], events.x[:events.i]] = events.p[:events.i]
        self.time += dt
        img = np.ones((self.height, self.width), dtype=np.float32) * 125
        ind = np.where(self.time_surface > 0)
        if len(ind[0]) > 0:
            decay = np.exp(-(self.time - self.time_surface[ind].astype(np.float32)) / self.tau)
            polarity_value = self.pol_surface[ind] * 2.0 - 1.0
            img[ind] = 125 + polarity_value * decay * 125
        img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
        return cv2.applyColorMap(img_uint8, cv2.COLORMAP_VIRIDIS)

# --- RL: Modified main rendering function for RL integration ---
def run_simulation(xml_path, obj_name, agent=None, reference_memory=None, mode='display'):
    """
    Main simulation function, adapted for different modes.
    Modes:
    - 'display': Just shows the rotating object (original behavior).
    - 'reference': Builds and returns a memory of a reference object.
    - 'train': Runs the RL loop to train the agent.
    - 'inference': Uses learned policy to explore and returns memory.
    - 'demo': Interactive mode with manual rotation control via arrow keys.
    """
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, HEIGHT, WIDTH)
    camera_id = -1
    dvs = DvsSensor("RealTimeDVS")
    dvs.initCamera(WIDTH, HEIGHT, lat=LAT, jit=JIT, ref=REF, tau=TAU, th_pos=TH_POS, th_neg=TH_NEG, th_noise=TH_NOISE, bgnp=BGNP, bgnn=BGNN)
    
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    renderer.update_scene(data)
    first_frame = renderer.render()
    first_luv = cv2.cvtColor(first_frame, cv2.COLOR_RGB2LUV)
    first_lum = first_luv[:, :, 0] / 255.0 * 1e4
    dvs.init_image(first_lum)
    
    event_renderer = EventFrameRenderer(WIDTH, HEIGHT, tau=3*DT)
    
    # Init attention network and DINO
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    net_attention = initialise_attention(device, ATTENTION_PARAMS)
    transform = T.Compose([T.ToTensor()])
    dino_device = torch.device("cpu") # DINO on CPU
    dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', verbose=False).to(dino_device)
    dino_model.eval()
    dino_transform = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    working_memory = WorkingMemory(dino_model, dino_transform, dino_device, ssp_dim=SSP_DIM)
    
    window_name = f"RL Discrimination: {obj_name} (Mode: {mode})"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, WIDTH * 2, HEIGHT)

    yaw_angle, pitch_angle = 0.0, 0.0
    yaw_speed, pitch_speed = (INITIAL_YAW_SPEED, 0.0) if mode not in ['train', 'inference'] else (0.0, 0.0)

    # TODO: use more ref steps (e.g. 2000) if it works and we can run it on a GPU
    max_steps = 200 if mode in ['reference', 'inference'] else 200 # Shorter for ref, longer for training
    if mode == 'demo':
        max_steps = 100000  # Run indefinitely until user quits

    for step in range(max_steps):
        # --- RL: State, Action, Reward, Next State ---
        current_state = None
        saccade_target = None
        top_k_coords_normalized = np.zeros(K_SALIENCY * 2)

        # 1. GENERATE FRAME AND EVENTS
        yaw_angle += yaw_speed
        pitch_angle += pitch_speed
        mujoco.mj_resetData(model, data)
        # Combine quaternions for ZYX rotation
        cy, sy = np.cos(yaw_angle * 0.5), np.sin(yaw_angle * 0.5)
        cp, sp = np.cos(pitch_angle * 0.5), np.sin(pitch_angle * 0.5)
        data.qpos[3] = cy * cp  # qw
        data.qpos[4] = 0        # qx
        data.qpos[5] = cy * sp  # qy
        data.qpos[6] = sy * cp  # qz
        mujoco.mj_forward(model, data)
        renderer.update_scene(data)
        pixels = renderer.render()
        pixels_luv = cv2.cvtColor(pixels, cv2.COLOR_RGB2LUV)
        luminance = pixels_luv[:, :, 0] / 255.0 * 1e4
        events = dvs.update(luminance, DT)
        event_frame = event_renderer.update(events, DT)

        # 2. RUN ATTENTION
        saliency_map, salmax_coords = None, None
        top_k_coords = []
        if events.i > 0:
            event_gray = cv2.cvtColor(event_frame, cv2.COLOR_BGR2GRAY)
            event_tensor = transform(event_gray)
            saliency_map, _ = run_attention(event_tensor, net_attention, device, (HEIGHT, WIDTH), ATTENTION_PARAMS['num_pyr'])
            top_k_coords = get_top_k_saliency_coords(saliency_map, K_SALIENCY)
            if top_k_coords:
                salmax_coords = top_k_coords[0] # Default to top-1

        # --- RL: Main Logic ---
        if mode in ['train', 'inference'] and agent is not None:
            # 3. CONSTRUCT STATE
            current_mem = working_memory.get_memory()
            if top_k_coords:
                # Normalize coordinates to be between -1 and 1
                coords_flat = np.array(top_k_coords, dtype=np.float32).flatten()
                coords_flat[::2] = (coords_flat[::2] / HEIGHT) * 2 - 1 # Y
                coords_flat[1::2] = (coords_flat[1::2] / WIDTH) * 2 - 1 # X
                top_k_coords_normalized[:len(coords_flat)] = coords_flat

            current_state = np.concatenate([current_mem, top_k_coords_normalized])
            if mode == 'train':
                sim_before = F.cosine_similarity(torch.Tensor(reference_memory), torch.Tensor(current_mem), dim=0).item()

            # 4. CHOOSE AND EXECUTE ACTION
            if mode == 'train':
                action_idx = agent.choose_action(current_state)
            else:  # inference mode - use greedy policy
                action_idx = agent.choose_action_greedy(current_state)

            if action_idx < NUM_MOVE_ACTIONS:
                # It's a move action
                if action_idx == 0: yaw_speed += SPEED_INCREMENT
                elif action_idx == 1: yaw_speed -= SPEED_INCREMENT
                elif action_idx == 2: pitch_speed += SPEED_INCREMENT
                elif action_idx == 3: pitch_speed -= SPEED_INCREMENT
                # action_idx == 4 is do_nothing
                saccade_target = salmax_coords # Default saccade
            else:
                # It's a saccade action
                saccade_idx = action_idx - NUM_MOVE_ACTIONS
                if top_k_coords and saccade_idx < len(top_k_coords):
                    saccade_target = top_k_coords[saccade_idx]
                else:
                    saccade_target = salmax_coords # Fallback
        else: # For 'reference' or 'display' mode
            saccade_target = salmax_coords

        # 5. PROCESS SACCADE AND GET NEXT STATE
        roi_coords = None
        if saccade_target:
            y, x = saccade_target[0], saccade_target[1]
            x1, y1 = max(x - (ROI_SIZE // 2), 0), max(y - (ROI_SIZE // 2), 0)
            x2, y2 = min(x1 + ROI_SIZE, WIDTH), min(y1 + ROI_SIZE, HEIGHT)
            roi_coords = (x1, y1, x2, y2)
            image_patch = event_frame[y1:y2, x1:x2]
            if image_patch.shape[0] > 0 and image_patch.shape[1] > 0:
                rotation_state = {'quaternion': (data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6])}
                working_memory.process_saccade(image_patch, (x, y), rotation_state)

        if mode == 'train' and agent is not None and current_state is not None:
            # 6. CALCULATE REWARD & STORE EXPERIENCE
            next_mem = working_memory.get_memory()
            sim_after = F.cosine_similarity(torch.Tensor(reference_memory), torch.Tensor(next_mem), dim=0).item()
            reward = sim_before - sim_after # Reward for *reducing* similarity

            next_state = np.concatenate([next_mem, top_k_coords_normalized]) # Saliency map is from current step
            agent.remember(current_state, action_idx, reward, next_state)
            
            # 7. TRAIN AGENT
            loss = agent.replay()
            if step % TARGET_UPDATE_FREQ == 0:
                agent.update_target_net()
                if loss is not None:
                    print(f"Step {step}/{max_steps} | Loss: {loss:.4f} | Reward: {reward:.4f} | Epsilon: {agent.steps_done}")


        # 6. VISUALIZE
        display_original = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        event_frame_display = event_frame.copy()
        if saccade_target:
             cv2.drawMarker(event_frame_display, (saccade_target[1], saccade_target[0]), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
        if roi_coords:
             cv2.rectangle(display_original, (roi_coords[0], roi_coords[1]), (roi_coords[2], roi_coords[3]), (0, 255, 0), 2)
        combined = np.hstack([display_original, event_frame_display])
        cv2.putText(combined, f"Mode: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if mode == 'demo':
            cv2.putText(combined, f"Yaw: {yaw_speed:.3f} | Pitch: {pitch_speed:.3f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(combined, "Arrow Keys: Yaw/Pitch | Q: Quit", (10, HEIGHT - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow(window_name, combined)

        # Handle keyboard input
        if mode == 'demo':
            key = cv2.waitKeyEx(1)  # waitKeyEx gives better arrow key support
        else:
            key = cv2.waitKey(1)

        if key == ord('q') or key == ord('Q'):
            break
        elif mode == 'demo' and key != -1:
            # Arrow keys for adjusting rotation speeds
            # Arrow key codes: Up=2490368, Down=2621440, Left=2424832, Right=2555904 (common values)
            # Also handle: Up=82/0, Down=84/1, Left=81/2, Right=83/3 (alternative codes)
            if key == 2490368 or key == 82 or key == 0:  # Up arrow
                pitch_speed += SPEED_INCREMENT
                print(f"Pitch speed increased to {pitch_speed:.3f}")
            elif key == 2621440 or key == 84 or key == 1:  # Down arrow
                pitch_speed -= SPEED_INCREMENT
                print(f"Pitch speed decreased to {pitch_speed:.3f}")
            elif key == 2424832 or key == 81 or key == 2:  # Left arrow
                yaw_speed -= SPEED_INCREMENT
                print(f"Yaw speed decreased to {yaw_speed:.3f}")
            elif key == 2555904 or key == 83 or key == 3:  # Right arrow
                yaw_speed += SPEED_INCREMENT
                print(f"Yaw speed increased to {yaw_speed:.3f}")

    cv2.destroyWindow(window_name)
    renderer.close()

    if mode in ['reference', 'inference']:
        return working_memory.get_memory()
    return None


def save_memories(memories_dict, output_dir='memories'):
    """Save object memories to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"memories_{timestamp}.json")

    # Convert numpy arrays to lists for JSON serialization
    serializable_dict = {}
    for obj_name, memory in memories_dict.items():
        serializable_dict[obj_name] = memory.tolist() if isinstance(memory, np.ndarray) else memory

    with open(filename, 'w') as f:
        json.dump(serializable_dict, f)

    print(f"✓ Memories saved to {filename}")
    return filename


def main():
    parser = argparse.ArgumentParser(description='Run RL agent for object discrimination.')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference', 'demo'], help='Mode: train, inference, or demo.')
    parser.add_argument('--ref', type=str, default='dog', help='Name of the reference object.')
    parser.add_argument('--targets', type=str, nargs='+', default=None, help='Names of target objects to discriminate (space-separated). If not provided, uses all objects except reference.')
    parser.add_argument('--num_episodes', type=int, default=10, help='Number of training episodes (each episode randomly selects a target object).')
    parser.add_argument('--agent_path', type=str, default='agent.pt', help='Path to save/load agent weights.')
    parser.add_argument('--memory_dir', type=str, default='memories', help='Directory to save memories.')
    parser.add_argument('--demo_obj', type=str, default='dog', help='Object to display in demo mode.')
    args = parser.parse_args()

    # --- RL: Define state and action dimensions ---
    state_dim = SSP_DIM + (K_SALIENCY * 2) # Memory vector + K coordinates (y,x)

    # --- RL: Instantiate the agent ---
    agent = DiscriminationAgent(state_dim, TOTAL_ACTIONS)

    if args.mode == 'train':
        # --- PHASE 1: Build reference memory ---
        print("\n" + "="*80)
        print(f"PHASE 1: Building reference memory for '{args.ref}'...")
        ref_xml_path = os.path.join(OBJECTS_DIR, args.ref, f"{args.ref}.xml")
        if not os.path.exists(ref_xml_path):
            print(f"Error: Reference object '{args.ref}' not found."); return

        reference_memory = run_simulation(ref_xml_path, args.ref, mode='reference')
        print(f"✓ Reference memory for '{args.ref}' created.")
        print("="*80 + "\n")

        # --- Discover or validate target objects ---
        if args.targets is None:
            # Auto-discover all objects except reference
            all_objects = [d for d in os.listdir(OBJECTS_DIR) if os.path.isdir(os.path.join(OBJECTS_DIR, d)) and d != args.ref]
            targets = all_objects
            print(f"Auto-discovered {len(targets)} target objects: {targets}")
        else:
            targets = args.targets
            print(f"Using specified target objects: {targets}")

        # Validate all target objects exist
        target_xml_paths = {}
        for obj in targets:
            xml_path = os.path.join(OBJECTS_DIR, obj, f"{obj}.xml")
            if not os.path.exists(xml_path):
                print(f"Warning: Target object '{obj}' not found, skipping.")
            else:
                target_xml_paths[obj] = xml_path

        if not target_xml_paths:
            print("Error: No valid target objects found."); return

        print(f"Training on {len(target_xml_paths)} objects: {list(target_xml_paths.keys())}")
        print("="*80 + "\n")

        # --- PHASE 2: Train agent on multiple target objects ---
        print("="*80)
        print(f"PHASE 2: Training agent for {args.num_episodes} episodes...")
        print(f"Each episode randomly selects from: {list(target_xml_paths.keys())}")
        print("="*80 + "\n")

        for episode in range(args.num_episodes):
            # Randomly select a target object for this episode
            target_obj = random.choice(list(target_xml_paths.keys()))
            target_xml_path = target_xml_paths[target_obj]

            print(f"\n--- Episode {episode + 1}/{args.num_episodes}: Training on '{target_obj}' ---")
            run_simulation(target_xml_path, target_obj, agent=agent, reference_memory=reference_memory, mode='train')
            print(f"✓ Episode {episode + 1} complete!")

        print("\n" + "="*80)
        print("✓ All training complete!")
        print("="*80)

        # Save trained agent
        agent.save(args.agent_path)

    elif args.mode == 'inference':
        # Load trained agent
        if not os.path.exists(args.agent_path):
            print(f"Error: Agent checkpoint '{args.agent_path}' not found."); return
        agent.load(args.agent_path)

        # --- Discover or validate target objects ---
        if args.targets is None:
            # Auto-discover all objects (including reference)
            all_objects = [d for d in os.listdir(OBJECTS_DIR) if os.path.isdir(os.path.join(OBJECTS_DIR, d))]
            targets = all_objects
            print(f"Auto-discovered {len(targets)} objects: {targets}")
        else:
            targets = args.targets
            print(f"Using specified objects: {targets}")

        # Validate all objects exist
        object_xml_paths = {}
        for obj in targets:
            xml_path = os.path.join(OBJECTS_DIR, obj, f"{obj}.xml")
            if not os.path.exists(xml_path):
                print(f"Warning: Object '{obj}' not found, skipping.")
            else:
                object_xml_paths[obj] = xml_path

        if not object_xml_paths:
            print("Error: No valid objects found."); return

        print(f"\n" + "="*80)
        print(f"INFERENCE MODE: Generating memories for {len(object_xml_paths)} objects")
        print("="*80 + "\n")

        # Generate memories for all objects
        memories = {}
        for obj_name, xml_path in object_xml_paths.items():
            print(f"Processing '{obj_name}'...")
            memory = run_simulation(xml_path, obj_name, agent=agent, mode='inference')
            memories[obj_name] = memory
            print(f"✓ Memory for '{obj_name}' generated.\n")

        print("="*80)
        print("✓ All memories generated!")
        print("="*80 + "\n")

        # Save memories
        save_memories(memories, output_dir=args.memory_dir)

    elif args.mode == 'demo':
        # --- DEMO MODE: Manual control of object rotation ---
        print("\n" + "="*80)
        print(f"DEMO MODE: Viewing '{args.demo_obj}'")
        print("="*80)
        print("Controls:")
        print("  Arrow Keys: Adjust yaw (left/right) and pitch (up/down) speeds")
        print("  Q: Quit")
        print("="*80 + "\n")

        demo_xml_path = os.path.join(OBJECTS_DIR, args.demo_obj, f"{args.demo_obj}.xml")
        if not os.path.exists(demo_xml_path):
            print(f"Error: Demo object '{args.demo_obj}' not found."); return

        run_simulation(demo_xml_path, args.demo_obj, mode='demo')

        print("\n" + "="*80)
        print("✓ Demo complete!")
        print("="*80)


if __name__ == "__main__":
    main()

