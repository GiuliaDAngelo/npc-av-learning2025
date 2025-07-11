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
import matplotlib.pyplot as plt

import sspspace
from transformers import CLIPProcessor, CLIPModel


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

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
config = Config()

#### I need to mask the hand in the RGB frames
# surrs = ['s2', 's3', 's4', 's5', 's6']
surrs = ['s6']

for surr in surrs:

    # path_data = '/Users/giuliadangelo/workspace/data/DATASETs/core50/350x350evframes/'
    path_data = '/Users/giuliadangelo/Downloads/npc-av-learning/core50cropped/evframes/'+surr
    hand_mask_path = '/Users/giuliadangelo/Downloads/npc-av-learning/core50cropped/rgbframes/'+surr

    scenario = natsorted([d for d in os.listdir(path_data) if os.path.isdir(os.path.join(path_data, d))])
    #sensor
    max_x, max_y = 128, 128
    resolution = (max_y, max_x)
    box_size = 20
    #oms
    size_krn_after_oms = 121
    OMS = np.zeros((size_krn_after_oms, size_krn_after_oms), dtype=np.float32)
    vSliceOMS = torch.zeros((1, size_krn_after_oms, size_krn_after_oms), dtype=torch.float32).to(device)
    #saliency map
    saliency_map = np.zeros((max_y, max_x), dtype=np.float32)
    salmax_coords = np.zeros((2,), dtype=np.int32)
    #encoder
    coord_encoder = sspspace.RandomSSPSpace(domain_dim=2, ssp_dim=512)


    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()


    for objects_i in scenario:
        object_memory = coord_encoder.encode([[0, 0]])

        # Initializing networks
        net_center, net_surround = initialize_oms(device, config.OMS_PARAMS)
        net_attention = initialise_attention(device, config.ATTENTION_PARAMS)
        obj_path_data = os.path.join(path_data, objects_i)
        data_files = natsorted([f for f in os.listdir(obj_path_data) if os.path.isfile(os.path.join(obj_path_data, f))])

        for data_file_i in data_files:
            img_path = os.path.join(obj_path_data, data_file_i)
            if int(data_file_i.split('.')[0]) < 6:
               pass
            else:
                hand_mask_filename = f"C_0{int(surr.strip('/').replace('s', ''))}_{int(objects_i.split('o')[1]):02d}_{int(data_file_i.split('.')[0]):03d}-pred.png"
                img = Image.open(img_path)
                hand_mask_img = Image.open(os.path.join(hand_mask_path, objects_i, hand_mask_filename))
                window = transform(img)
                hand_mask = transform(hand_mask_img)
                hand_mask = torch.where(hand_mask != 0.0, torch.tensor(255, dtype=hand_mask.dtype), hand_mask)
                window_original = window
                window = window * (hand_mask == 0)

                # computing egomotion
                wOMS = torch.tensor(window, dtype=torch.float32).to(device)
                OMS, indexes = egomotion(wOMS, net_center, net_surround, device, config.MAX_Y,
                                         config.MAX_X, config.OMS_PARAMS['threshold'])

                vSliceOMS[:] = OMS.squeeze(0)
                # #window
                # saliency_map[:], salmax_coords[:] = run_attention(
                #     window, net_attention, device, resolution, config.ATTENTION_PARAMS['num_pyr']
                # )
                #OMS
                saliency_map[:], salmax_coords[:] = run_attention(
                    vSliceOMS[:], net_attention, device, resolution, config.ATTENTION_PARAMS['num_pyr']
                )

                fig, ax = plt.subplots(figsize=(4, 4))
                cax = ax.imshow(saliency_map, cmap='jet', vmin=0, vmax=255)
                ax.scatter(salmax_coords[1], salmax_coords[0], s=60, c='white', edgecolors='black', linewidths=1)
                # ax.set_title('Saliency')
                ax.axis('off')
                fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04, label='Saliency (0–255)')

                fig.canvas.draw()
                plt.close(fig)

                # --- Events image with bounding box (right) ---
                window_img = window.detach().cpu().numpy().squeeze(0)
                window_img = (window_img * 255).clip(0, 255).astype(np.uint8)
                window_img_color = cv2.cvtColor(window_img, cv2.COLOR_GRAY2BGR)


                window_original = window_original.detach().cpu().numpy().squeeze(0)
                window_original = (window_original * 255).clip(0, 255).astype(np.uint8)
                window_original_colour = cv2.cvtColor(window_original, cv2.COLOR_GRAY2BGR)


                x, y = salmax_coords[1], salmax_coords[0]
                x1, y1 = max(x - box_size//2, 0), max(y - box_size//2, 0)
                x2, y2 = min(x + box_size//2, window_img.shape[1]), min(y + box_size//2, window_img.shape[0])

                cv2.rectangle(window_img_color, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(window_img_color, 'Events', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

                mask = np.zeros_like(window_img_color)
                mask[y1:y2, x1:x2] = window_img_color[y1:y2, x1:x2]
                window_img_boxed = mask

                roi_center = coord_encoder.encode([[x, y]])
                current_roi = clip_processor(images=[window_img_color[y1:y2, x1:x2]], return_tensors="pt", padding=True).to(device)
                with torch.no_grad():
                    image_features = clip_model.get_image_features(**current_roi)

                img_feat_ssp = sspspace.SSP(image_features.cpu().numpy())
                new_roi = roi_center * img_feat_ssp
                gamma = 0.99
                object_memory = gamma * object_memory + (1 - gamma) * new_roi


                # --- VISUALISATION --- #
                window_img_colorized = cv2.cvtColor(window_img, cv2.COLOR_GRAY2BGR)
                OMS = OMS.squeeze(0).squeeze(0).cpu().detach().numpy()
                OMS_map_color = cv2.applyColorMap(OMS.astype(np.uint8), cv2.COLORMAP_JET)
                saliency_map_color = cv2.applyColorMap(saliency_map.astype(np.uint8), cv2.COLORMAP_JET)

                #resize images to match dimensions
                target_height = saliency_map_color.shape[0]
                target_width = saliency_map_color.shape[1]
                OMS_map_color_resized = cv2.resize(OMS_map_color, (target_width, target_height))
                window_img_boxed_resized = cv2.resize(window_img_boxed, (target_width, target_height))

                combined = np.hstack((window_original_colour, window_img_colorized, OMS_map_color_resized, saliency_map_color, window_img_boxed_resized))

                cv2.imshow('Window | OMS | Saliency | Atention', combined)
                cv2.waitKey(1)

                # Free memory
                del window
                torch.mps.empty_cache()

                # Reset
                window = torch.zeros((1, max_y, max_x), dtype=torch.float32)
                saliency_map = np.zeros((max_y, max_x), dtype=np.float32)

        np.save(
            f'/Users/giuliadangelo/Downloads/npc-av-learning/core50cropped/workingmemory/{surr}/object_{objects_i}_memory.npy',
            object_memory)
    cv2.destroyAllWindows()
