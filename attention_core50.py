import numpy as np
from attention_helpers import initialise_attention, run_attention
import torch
import cv2
import os
from natsort import natsorted
from PIL import Image
import torchvision.transforms as T

import matplotlib
matplotlib.use('Agg')  # Use Agg backend (no GUI)
import matplotlib.pyplot as plt

transform = T.Compose([
    T.Grayscale(),
    T.ToTensor(),
])

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

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
config = Config()

path_data = '/Users/giuliadangelo/workspace/data/DATASETs/core50/350x350evframes/'


scenario = natsorted([d for d in os.listdir(path_data) if os.path.isdir(os.path.join(path_data, d))])
max_x, max_y = 350, 350
resolution = (max_y, max_x)
box_size = 40

saliency_map = np.zeros((max_y, max_x), dtype=np.float32)
salmax_coords = np.zeros((2,), dtype=np.int32)

for scenario_i in scenario:
    obj_path = os.path.join(path_data, scenario_i)
    objects = natsorted([d for d in os.listdir(obj_path) if os.path.isdir(os.path.join(obj_path, d))])
    for objects_i in objects:
        net_attention = initialise_attention(device, config.ATTENTION_PARAMS)
        obj_path_data = os.path.join(obj_path, objects_i)
        data_files = natsorted([f for f in os.listdir(obj_path_data) if os.path.isfile(os.path.join(obj_path_data, f))])
        for data_file_i in data_files:
            img_path = os.path.join(obj_path_data, data_file_i)
            img = Image.open(img_path)
            window = transform(img).unsqueeze(0).squeeze(1)
            vals, counts = torch.unique(window, return_counts=True)
            max_mode = vals[torch.argmax(counts)]
            window = torch.where(window == max_mode, torch.tensor(0.0, device=window.device),
                                 torch.tensor(255.0, device=window.device))

            saliency_map[:], salmax_coords[:] = run_attention(
                window, net_attention, device, resolution, config.ATTENTION_PARAMS['num_pyr']
            )

            fig, ax = plt.subplots(figsize=(4, 4))
            cax = ax.imshow(saliency_map, cmap='jet', vmin=0, vmax=255)
            ax.scatter(salmax_coords[1], salmax_coords[0], s=60, c='white', edgecolors='black', linewidths=1)
            ax.set_title('Saliency')
            ax.axis('off')
            fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04, label='Saliency (0–255)')

            fig.canvas.draw()
            plt.close(fig)


            # --- Events image with bounding box (right) ---
            window_img = window.detach().cpu().numpy().squeeze(0)
            window_img = (window_img * 255).clip(0, 255).astype(np.uint8)
            window_img_color = cv2.cvtColor(window_img, cv2.COLOR_GRAY2BGR)

            x, y = salmax_coords[1], salmax_coords[0]
            x1, y1 = max(x - box_size//2, 0), max(y - box_size//2, 0)
            x2, y2 = min(x + box_size//2, window_img.shape[1]), min(y + box_size//2, window_img.shape[0])

            cv2.rectangle(window_img_color, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(window_img_color, 'Events', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

            mask = np.zeros_like(window_img_color)
            mask[y1:y2, x1:x2] = window_img_color[y1:y2, x1:x2]
            window_img_boxed = mask

            # --- Concatenate and show ---
            saliency_map_color = cv2.applyColorMap(saliency_map.astype(np.uint8), cv2.COLORMAP_JET)
            combined = np.hstack((saliency_map_color, window_img_boxed))
            cv2.imshow('Saliency (left) & Events in Box (right)', combined)
            cv2.waitKey(1)

            # Free memory
            del window
            torch.mps.empty_cache()

            # Reset
            window = torch.zeros((1, max_y, max_x), dtype=torch.float32)
            saliency_map = np.zeros((max_y, max_x), dtype=np.float32)

cv2.destroyAllWindows()
