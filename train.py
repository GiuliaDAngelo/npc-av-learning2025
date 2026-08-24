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
import re
from datetime import datetime

# Add IEBCS to path
sys.path.append("IEBCS/src")
from dvs_sensor import DvsSensor
from dvs_sensor_torch import TorchDvsSensor
from event_buffer import EventBuffer

# Import attention mechanism
from attention_helpers import initialise_attention, run_attention
from itti_saliency import IttiKochSaliency

# Configuration
OBJECTS_DIR = "CRIB Data/mujoco_objects"
WIDTH = 640
HEIGHT = 480
CAMERA_NAME = None  # Use default camera

# Rotation control parameters (module-level so episode recorders can randomize
# the starting pose/trajectory for exploration diversity)
INITIAL_YAW_ANGLE = 0.0
INITIAL_PITCH_ANGLE = 0.0
INITIAL_YAW_SPEED = 0.05
INITIAL_PITCH_SPEED = 0.0
SPEED_INCREMENT = 0.01
MIN_SPEED = 0.01  # Minimum absolute speed to prevent static objects

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

# What the encoder sees at each fixation ('rgb' or 'events') and what drives the
# saliency map ('events': spiking VM attention on event frames, 'rgb': the same
# attention on the RGB grayscale, 'itti': Itti-Koch-Niebur on the RGB render).
# Defaults: encode appearance from the RGB render, use events only to find where
# to look. Overridable via --patch_source/--saliency_source.
PATCH_SOURCE = 'rgb'
SALIENCY_SOURCE = 'itti'
# When True, WorkingMemory keeps a raw per-saccade log (patch, coords, quaternion,
# embedding); run_simulation exposes the last episode's log as LAST_SACCADE_LOG.
RECORD_SACCADES = False
LAST_SACCADE_LOG = None

# --- RL: New constants for the Deep Q-Learning Agent ---
K_SALIENCY = 5              # Agent can choose from top K salient points
NUM_MOVE_ACTIONS = 5        # inc/dec yaw, inc/dec pitch, do_nothing
NUM_SACCADE_ACTIONS = K_SALIENCY
TOTAL_ACTIONS = NUM_MOVE_ACTIONS + NUM_SACCADE_ACTIONS
SSP_DIM = 384              # the same as small DINOv2 embedding size

# RL Hyperparameters
REPLAY_BUFFER_SIZE = 5000
BATCH_SIZE = 8
GAMMA_RL = 0.99             # RL discount factor
EPSILON_START = 0.9
EPSILON_END = 0.05
EPSILON_DECAY = 1000
TARGET_UPDATE_FREQ = 10     # Update target network every 10 episodes/steps
MAX_REF_STEPS = 200         # Shorter for reference/inference
# TODO: use more ref steps (e.g. 2000) if it works and we can run it on a GPU
MAX_TRAIN_STEPS = 500       # Longer for training

def enforce_minimum_speed(yaw_speed, pitch_speed, min_speed=MIN_SPEED):
    """
    Enforce minimum rotation speed to prevent static objects.
    If both speeds are too small, boost at least one to the minimum.
    """
    yaw_mag = abs(yaw_speed)
    pitch_mag = abs(pitch_speed)

    # If both speeds are below minimum, ensure at least one meets the minimum
    if yaw_mag < min_speed and pitch_mag < min_speed:
        # Boost the larger one (or yaw if equal)
        if yaw_mag >= pitch_mag:
            yaw_speed = min_speed if yaw_speed >= 0 else -min_speed
        else:
            pitch_speed = min_speed if pitch_speed >= 0 else -min_speed

    return yaw_speed, pitch_speed


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
        self.record = False
        self.saccade_log = []
        print(f"  Working Memory initialized (SSP dim: {ssp_dim})")

    def bind(self, a, b):
        # Newer sspspace wraps vectors in an SSP object (array in .v); unwrap for FFT binding
        a = np.atleast_2d(getattr(a, 'v', a))
        b = np.atleast_2d(getattr(b, 'v', b))
        return np.fft.ifft(np.fft.fft(a, axis=1) * np.fft.fft(b, axis=1), axis=1).real

    def process_saccade(self, image_patch, saccade_center, rotation_state):
        image_patch_rgb = cv2.cvtColor(image_patch, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_patch_rgb)
        input_tensor = self.dino_transform(pil_image).unsqueeze(0).to(self.dino_device)
        with torch.no_grad():
            # DINO embedding (small model) is 1x384
            dino_embedding = self.dino_model(input_tensor).cpu().numpy()

        if self.record:
            self.saccade_log.append({
                'patch': image_patch.copy(),          # BGR uint8, as cropped
                'coord': tuple(saccade_center),
                'quat': tuple(rotation_state['quaternion']),
                'embedding': dino_embedding.flatten().copy(),  # raw encoder output
            })

        if dino_embedding.shape[1] < self.ssp_dim:
            padding = np.zeros((1, self.ssp_dim - dino_embedding.shape[1]))
            dino_embedding = np.hstack([dino_embedding, padding])
        else:
            dino_embedding = dino_embedding[:, :self.ssp_dim]

        # normalize dino embedding so its norm is 1
        dino_embedding /= np.linalg.norm(dino_embedding) + 1e-10
        
        x, y = saccade_center
        coord_ssp = self.coord_encoder.encode([[x, y]])
        quat = rotation_state['quaternion']
        quat_ssp = self.quat_encoder.encode([[quat[0], quat[1], quat[2], quat[3]]])
        # TODO: is this the correct way to bind 3 things?
        bound_img_coord = self.bind(dino_embedding, coord_ssp)
        bound_representation = self.bind(bound_img_coord, quat_ssp)
        # TODO: can we do something more sophisticated than simple exponential moving average?
        #       sparse 3D representation by storing the different (protypical) views separately?
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
def run_simulation(xml_path, obj_name, agent=None, reference_memory=None, mode='display',
                   obj_class=None, memory_classes=None):
    """
    Main simulation function, adapted for different modes.
    Modes:
    - 'display': Just shows the rotating object (original behavior).
    - 'reference': Builds and returns a memory of a reference object (uses bottom-up saccades).
    - 'train': Runs the RL loop to train the agent using contrastive learning.
    - 'inference': Uses learned policy to explore and returns memory.
    - 'demo': Interactive mode with manual rotation control via arrow keys.

    Args:
        reference_memory: List of memory arrays from all previously seen objects.
        obj_class: Class label for current object (for contrastive learning).
        memory_classes: List of class labels corresponding to reference_memory arrays.
    """
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, HEIGHT, WIDTH)
    camera_id = -1
    dvs = TorchDvsSensor("RealTimeDVS")  # GPU port of the IEBCS sensor (see dvs_sensor_torch.py)
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
    itti_model = IttiKochSaliency(device) if SALIENCY_SOURCE == 'itti' else None
    transform = T.Compose([T.ToTensor()])
    dino_device = device # DINO on the same accelerator as the attention net
    dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', verbose=False).to(dino_device)
    dino_model.eval()
    dino_transform = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    working_memory = WorkingMemory(dino_model, dino_transform, dino_device, ssp_dim=SSP_DIM)
    working_memory.record = RECORD_SACCADES
    
    window_name = f"RL Discrimination: {obj_name} (Mode: {mode})"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, WIDTH * 2, HEIGHT)

    yaw_angle, pitch_angle = INITIAL_YAW_ANGLE, INITIAL_PITCH_ANGLE
    yaw_speed, pitch_speed = INITIAL_YAW_SPEED, INITIAL_PITCH_SPEED

    # Enforce minimum speed from the start
    yaw_speed, pitch_speed = enforce_minimum_speed(yaw_speed, pitch_speed)

    max_steps = MAX_REF_STEPS if mode in ['reference', 'inference'] else MAX_TRAIN_STEPS
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

        # 2. RUN ATTENTION (RGB-based saliency does not depend on event activity)
        saliency_map, salmax_coords = None, None
        top_k_coords = []
        if events.i > 0 or SALIENCY_SOURCE in ('rgb', 'itti'):
            if SALIENCY_SOURCE == 'itti':
                saliency_map = itti_model.compute(pixels)
            else:
                if SALIENCY_SOURCE == 'rgb':
                    attn_gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
                else:
                    attn_gray = cv2.cvtColor(event_frame, cv2.COLOR_BGR2GRAY)
                event_tensor = transform(attn_gray)
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
                # Contrastive learning: separate same-class from different-class memories
                if isinstance(reference_memory, list) and memory_classes is not None:
                    same_class_sims_before = []
                    other_class_sims_before = []

                    for ref_mem, ref_class in zip(reference_memory, memory_classes):
                        sim = F.cosine_similarity(torch.Tensor(ref_mem), torch.Tensor(current_mem), dim=0).item()
                        if ref_class == obj_class:
                            same_class_sims_before.append(sim)
                        else:
                            other_class_sims_before.append(sim)

                    # Store for later reward calculation
                    sim_to_same_before = max(same_class_sims_before) if same_class_sims_before else 0.0
                    sim_to_other_before = max(other_class_sims_before) if other_class_sims_before else 0.0
                else:
                    # Fallback for non-contrastive mode
                    if isinstance(reference_memory, list):
                        similarities_before = [F.cosine_similarity(torch.Tensor(ref_mem), torch.Tensor(current_mem), dim=0).item()
                                              for ref_mem in reference_memory]
                        sim_before = max(similarities_before)
                    else:
                        sim_before = F.cosine_similarity(torch.Tensor(reference_memory), torch.Tensor(current_mem), dim=0).item()
                    sim_to_same_before = 0.0
                    sim_to_other_before = sim_before

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

                # Enforce minimum speed to prevent static objects
                yaw_speed, pitch_speed = enforce_minimum_speed(yaw_speed, pitch_speed)
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
            # TODO: do we want to incorporate the periphery
            #       maybe downsample the region outside the ROI?
            if PATCH_SOURCE == 'rgb':
                # Encode appearance from the RGB render; events only guided the saccade.
                # (BGR, so process_saccade's BGR->RGB conversion stays correct.)
                image_patch = cv2.cvtColor(pixels[y1:y2, x1:x2], cv2.COLOR_RGB2BGR)
            else:
                image_patch = event_frame[y1:y2, x1:x2]
            if image_patch.shape[0] > 0 and image_patch.shape[1] > 0:
                rotation_state = {'quaternion': (data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6])}
                working_memory.process_saccade(image_patch, (x, y), rotation_state)

        if mode == 'train' and agent is not None and current_state is not None:
            # 6. CALCULATE REWARD & STORE EXPERIENCE
            next_mem = working_memory.get_memory()

            # Contrastive learning reward
            if isinstance(reference_memory, list) and memory_classes is not None:
                same_class_sims_after = []
                other_class_sims_after = []

                for ref_mem, ref_class in zip(reference_memory, memory_classes):
                    sim = F.cosine_similarity(torch.Tensor(ref_mem), torch.Tensor(next_mem), dim=0).item()
                    if ref_class == obj_class:
                        same_class_sims_after.append(sim)
                    else:
                        other_class_sims_after.append(sim)

                sim_to_same_after = max(same_class_sims_after) if same_class_sims_after else 0.0
                sim_to_other_after = max(other_class_sims_after) if other_class_sims_after else 0.0

                # Reward: increase similarity to same class + decrease similarity to other classes
                reward_same = (sim_to_same_after - sim_to_same_before)  # Positive if becoming more similar to same class
                reward_other = (sim_to_other_before - sim_to_other_after)  # Positive if becoming less similar to other classes

                # Combine rewards
                reward = reward_same + reward_other
            else:
                # Fallback for non-contrastive mode
                if isinstance(reference_memory, list):
                    similarities_after = [F.cosine_similarity(torch.Tensor(ref_mem), torch.Tensor(next_mem), dim=0).item()
                                         for ref_mem in reference_memory]
                    sim_after = max(similarities_after)
                else:
                    sim_after = F.cosine_similarity(torch.Tensor(reference_memory), torch.Tensor(next_mem), dim=0).item()
                reward = sim_to_other_before - sim_after  # Reward for reducing similarity

            next_state = np.concatenate([next_mem, top_k_coords_normalized]) # Saliency map is from current step
            agent.remember(current_state, action_idx, reward, next_state)
            
            # 7. TRAIN AGENT
            loss = agent.replay()
            if step % TARGET_UPDATE_FREQ == 0:
                agent.update_target_net()
                if loss is not None:
                    # Enhanced logging for contrastive learning
                    if isinstance(reference_memory, list) and memory_classes is not None:
                        print(f"Step {step}/{max_steps} | Loss: {loss:.4f} | Reward: {reward:.4f} "
                              f"(same: {reward_same:.4f}, other: {reward_other:.4f}) | Epsilon: {agent.steps_done}")
                    else:
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
            # macOS: Up=63232, Down=63233, Left=63234, Right=63235
            # Linux/Windows: Up=2490368, Down=2621440, Left=2424832, Right=2555904
            # Alternative: Up=82/0, Down=84/1, Left=81/2, Right=83/3
            if key == 63232 or key == 2490368 or key == 82 or key == 0:  # Up arrow
                pitch_speed += SPEED_INCREMENT
                yaw_speed, pitch_speed = enforce_minimum_speed(yaw_speed, pitch_speed)
                print(f"Pitch speed increased to {pitch_speed:.3f}")
            elif key == 63233 or key == 2621440 or key == 84 or key == 1:  # Down arrow
                pitch_speed -= SPEED_INCREMENT
                yaw_speed, pitch_speed = enforce_minimum_speed(yaw_speed, pitch_speed)
                print(f"Pitch speed decreased to {pitch_speed:.3f}")
            elif key == 63234 or key == 2424832 or key == 81 or key == 2:  # Left arrow
                yaw_speed -= SPEED_INCREMENT
                yaw_speed, pitch_speed = enforce_minimum_speed(yaw_speed, pitch_speed)
                print(f"Yaw speed decreased to {yaw_speed:.3f}")
            elif key == 63235 or key == 2555904 or key == 83 or key == 3:  # Right arrow
                yaw_speed += SPEED_INCREMENT
                yaw_speed, pitch_speed = enforce_minimum_speed(yaw_speed, pitch_speed)
                print(f"Yaw speed increased to {yaw_speed:.3f}")

    cv2.destroyWindow(window_name)
    renderer.close()

    global LAST_SACCADE_LOG
    LAST_SACCADE_LOG = working_memory.saccade_log

    if mode in ['reference', 'inference']:
        return working_memory.get_memory()
    return None


def save_memories(memories_dict, output_dir='memories', classes_dict=None):
    """Save object memories to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"memories_{timestamp}.json")

    # Convert numpy arrays to lists for JSON serialization
    data_to_save = {
        'memories': {},
        'classes': classes_dict if classes_dict else {}
    }

    for obj_name, memory in memories_dict.items():
        data_to_save['memories'][obj_name] = memory.tolist() if isinstance(memory, np.ndarray) else memory

    with open(filename, 'w') as f:
        json.dump(data_to_save, f, indent=2)

    print(f"✓ Memories saved to {filename}")
    return filename


def get_object_instance_key(object_name, instance_id):
    """
    Create a unique key for an object instance.
    Examples: ('dog', 0) -> 'dog_0', ('cat', 2) -> 'cat_2'
    """
    return f"{object_name}_{instance_id}"


def main():
    parser = argparse.ArgumentParser(description='Run RL agent for contrastive object discrimination.')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference', 'demo'], help='Mode: train, inference, or demo.')
    parser.add_argument('--objects', type=str, nargs='+', default=None, help='Names of objects to use (space-separated). Objects will be randomly sampled during training. If not provided, uses all available objects.')
    parser.add_argument('--num_episodes', type=int, default=10, help='Number of training episodes (random samples from --objects).')
    parser.add_argument('--agent_path', type=str, default='agent.pt', help='Path to save/load agent weights.')
    parser.add_argument('--memory_dir', type=str, default='memories', help='Directory to save memories.')
    parser.add_argument('--demo_obj', type=str, default='dog', help='Object to display in demo mode.')
    parser.add_argument('--patch_source', type=str, default='rgb', choices=['rgb', 'events'],
                        help='Image source for the encoded fixation patch.')
    parser.add_argument('--saliency_source', type=str, default='itti', choices=['events', 'rgb', 'itti'],
                        help='Saliency method: spiking VM attention on events or RGB, or Itti-Koch-Niebur on RGB.')
    args = parser.parse_args()

    global PATCH_SOURCE, SALIENCY_SOURCE
    PATCH_SOURCE = args.patch_source
    SALIENCY_SOURCE = args.saliency_source
    print(f"Patch source: {PATCH_SOURCE} | Saliency source: {SALIENCY_SOURCE}")

    # --- RL: Define state and action dimensions ---
    state_dim = SSP_DIM + (K_SALIENCY * 2) # Memory vector + K coordinates (y,x)

    # --- RL: Instantiate the agent ---
    agent = DiscriminationAgent(state_dim, TOTAL_ACTIONS)

    if args.mode == 'train':
        # --- Discover or validate objects ---
        if args.objects is None:
            # Auto-discover all objects
            all_objects = [d for d in os.listdir(OBJECTS_DIR) if os.path.isdir(os.path.join(OBJECTS_DIR, d))]
            objects = all_objects
            print(f"Auto-discovered {len(objects)} objects: {objects}")
        else:
            objects = args.objects
            print(f"Using specified objects: {objects}")

        # Validate all objects exist
        object_xml_paths = {}
        for obj in objects:
            xml_path = os.path.join(OBJECTS_DIR, obj, f"{obj}.xml")
            if not os.path.exists(xml_path):
                print(f"Warning: Object '{obj}' not found, skipping.")
            else:
                object_xml_paths[obj] = xml_path

        if not object_xml_paths:
            print("Error: No valid objects found."); return

        print(f"\n" + "="*80)
        print(f"TRAINING: Randomly sampling from {len(object_xml_paths)} objects for {args.num_episodes} episodes")
        print(f"Available objects: {list(object_xml_paths.keys())}")
        print("="*80 + "\n")

        # Dictionary to store all object instance memories and their classes
        object_memories = {}
        object_classes = {}
        instance_counts = {obj_name: 0 for obj_name in object_xml_paths.keys()}

        for episode in range(args.num_episodes):
            # Randomly select an object for this episode
            obj_name = random.choice(list(object_xml_paths.keys()))
            xml_path = object_xml_paths[obj_name]
            obj_class = obj_name  # The class is just the object name itself

            # Create unique instance key
            instance_key = get_object_instance_key(obj_name, instance_counts[obj_name])
            instance_counts[obj_name] += 1

            print("="*80)
            if episode == 0:
                # First episode: use reference mode (pure bottom-up saccades)
                print(f"EPISODE {episode + 1}/{args.num_episodes}: '{instance_key}' (class: {obj_class}, baseline)")
                print("="*80 + "\n")
                memory = run_simulation(xml_path, instance_key, mode='reference')
                object_memories[instance_key] = memory
                object_classes[instance_key] = obj_class
                print(f"\n✓ Baseline memory for '{instance_key}' created.")
            else:
                # Subsequent episodes: train agent using contrastive learning
                previous_instance_keys = list(object_memories.keys())
                same_class_instances = [key for key in previous_instance_keys if object_classes[key] == obj_class]
                other_class_instances = [key for key in previous_instance_keys if object_classes[key] != obj_class]
                unique_other_classes = set([object_classes[key] for key in other_class_instances])

                print(f"EPISODE {episode + 1}/{args.num_episodes}: '{instance_key}' (class: {obj_class})")
                if same_class_instances:
                    print(f"  Same class ({obj_class}) instances seen: {len(same_class_instances)}")
                if unique_other_classes:
                    print(f"  Different classes seen: {sorted(unique_other_classes)} ({len(other_class_instances)} instances)")
                print("="*80 + "\n")

                # Convert memories dict to lists for passing to run_simulation
                previous_memories = [object_memories[key] for key in previous_instance_keys]
                previous_classes = [object_classes[key] for key in previous_instance_keys]

                # Train on this object instance
                run_simulation(xml_path, instance_key, agent=agent,
                             reference_memory=previous_memories, mode='train',
                             obj_class=obj_class, memory_classes=previous_classes)

                # After training, generate final memory for this instance
                print(f"\nGenerating final memory for '{instance_key}'...")
                memory = run_simulation(xml_path, instance_key, agent=agent, mode='inference')
                object_memories[instance_key] = memory
                object_classes[instance_key] = obj_class
                print(f"✓ Memory for '{instance_key}' saved.")

            print("="*80 + "\n")

        print("\n" + "="*80)
        print("✓ All training complete!")
        print(f"\nFinal statistics:")
        print(f"  Total instances: {len(object_memories)}")
        print(f"  Unique classes: {len(set(object_classes.values()))}")
        for obj_name in sorted(object_xml_paths.keys()):
            count = instance_counts[obj_name]
            print(f"  {obj_name}: {count} instance(s)")
        print("="*80)

        # Save trained agent
        agent.save(args.agent_path)

        # Save all object memories with class information
        save_memories(object_memories, output_dir=args.memory_dir, classes_dict=object_classes)

    elif args.mode == 'inference':
        # Load trained agent
        if not os.path.exists(args.agent_path):
            print(f"Error: Agent checkpoint '{args.agent_path}' not found."); return
        agent.load(args.agent_path)

        # --- Discover or validate objects ---
        if args.objects is None:
            # Auto-discover all objects
            all_objects = [d for d in os.listdir(OBJECTS_DIR) if os.path.isdir(os.path.join(OBJECTS_DIR, d))]
            objects = all_objects
            print(f"Auto-discovered {len(objects)} objects: {objects}")
        else:
            objects = args.objects
            print(f"Using specified objects: {objects}")

        # Validate all objects exist
        object_xml_paths = {}
        for obj in objects:
            xml_path = os.path.join(OBJECTS_DIR, obj, f"{obj}.xml")
            if not os.path.exists(xml_path):
                print(f"Warning: Object '{obj}' not found, skipping.")
            else:
                object_xml_paths[obj] = xml_path

        if not object_xml_paths:
            print("Error: No valid objects found."); return

        print(f"\n" + "="*80)
        print(f"INFERENCE MODE: Randomly sampling from {len(object_xml_paths)} objects for {args.num_episodes} episodes")
        print(f"Available objects: {list(object_xml_paths.keys())}")
        print("="*80 + "\n")

        # Dictionary to store all object instance memories and their classes
        object_memories = {}
        object_classes = {}
        instance_counts = {obj_name: 0 for obj_name in object_xml_paths.keys()}

        for episode in range(args.num_episodes):
            # Randomly select an object for this episode
            obj_name = random.choice(list(object_xml_paths.keys()))
            xml_path = object_xml_paths[obj_name]
            obj_class = obj_name  # The class is just the object name itself

            # Create unique instance key
            instance_key = get_object_instance_key(obj_name, instance_counts[obj_name])
            instance_counts[obj_name] += 1

            print(f"Episode {episode + 1}/{args.num_episodes}: Generating memory for '{instance_key}'...")
            memory = run_simulation(xml_path, instance_key, agent=agent, mode='inference')
            object_memories[instance_key] = memory
            object_classes[instance_key] = obj_class
            print(f"✓ Memory for '{instance_key}' generated.\n")

        print("="*80)
        print("✓ All memories generated!")
        print(f"\nFinal statistics:")
        print(f"  Total instances: {len(object_memories)}")
        print(f"  Unique classes: {len(set(object_classes.values()))}")
        for obj_name in sorted(object_xml_paths.keys()):
            count = instance_counts[obj_name]
            print(f"  {obj_name}: {count} instance(s)")
        print("="*80 + "\n")

        # Save memories with class information
        save_memories(object_memories, output_dir=args.memory_dir, classes_dict=object_classes)

    elif args.mode == 'demo':
        # --- DEMO MODE: Manual control of object rotation ---
        # Use first object from --objects if provided, otherwise use --demo_obj
        demo_obj = args.objects[0] if args.objects else args.demo_obj

        print("\n" + "="*80)
        print(f"DEMO MODE: Viewing '{demo_obj}'")
        print("="*80)
        print("Controls:")
        print("  Arrow Keys: Adjust yaw (left/right) and pitch (up/down) speeds")
        print("  Q: Quit")
        print("="*80 + "\n")

        demo_xml_path = os.path.join(OBJECTS_DIR, demo_obj, f"{demo_obj}.xml")
        if not os.path.exists(demo_xml_path):
            print(f"Error: Demo object '{demo_obj}' not found."); return

        run_simulation(demo_xml_path, demo_obj, mode='demo')

        print("\n" + "="*80)
        print("✓ Demo complete!")
        print("="*80)


if __name__ == "__main__":
    main()

