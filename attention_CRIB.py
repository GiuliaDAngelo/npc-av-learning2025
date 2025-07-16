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
from training import CLIPEventClassifier  # remove if you are not training

import sspspace
from transformers import CLIPProcessor, CLIPModel

import torchvision.transforms as transforms

event_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])


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


bbox_savingFLAG = False
root = '/Users/giuliadangelo/Downloads/npc-av-learning/CRIB/train_data/'
path_data = root+'evframes/'
objects = natsorted([d for d in os.listdir(path_data) if os.path.isdir(os.path.join(path_data, d))])

for obj in objects:
    print(obj)
    #sensor
    max_x, max_y = 400, 400
    resolution = (max_y, max_x)
    box_size = 350
    #oms
    size_krn_after_oms = 393
    OMS = np.zeros((size_krn_after_oms, size_krn_after_oms), dtype=np.float32)
    vSliceOMS = torch.zeros((1, size_krn_after_oms, size_krn_after_oms), dtype=torch.float32).to(device)
    #saliency map
    saliency_map = np.zeros((max_y, max_x), dtype=np.float32)
    salmax_coords = np.zeros((2,), dtype=np.int32)
    #encoder
    coord_encoder = sspspace.RandomSSPSpace(domain_dim=2, ssp_dim=512)

    # clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    # clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    # clip_model.eval()

    NUM_CLASSES = 17  # Set to number of categories you trained on
    model_path = "clip_event_classifier.pth"

    event_clip_model = CLIPEventClassifier(num_classes=NUM_CLASSES).to(device)
    event_clip_model.load_state_dict(torch.load(model_path, map_location=device))
    event_clip_model.eval()

    net_center, net_surround = initialize_oms(device, config.OMS_PARAMS)
    net_attention = initialise_attention(device, config.ATTENTION_PARAMS)
    obj_path_data = os.path.join(path_data, obj)
    data_files = natsorted([f for f in os.listdir(obj_path_data) if os.path.isfile(os.path.join(obj_path_data, f))])

    object_memory = coord_encoder.encode([[0, 0]])
    bbox_saving_path = root + 'bbox/'
    os.mkdir(bbox_saving_path+obj) if not os.path.exists(bbox_saving_path + obj) else None
    for data_file_i in data_files:
        img_path = os.path.join(obj_path_data, data_file_i)

        img = Image.open(img_path)
        window = transform(img)
        window_original = window

        # computing egomotion
        wOMS = torch.tensor(window, dtype=torch.float32).to(device)
        OMS, indexes = egomotion(wOMS, net_center, net_surround, device, config.MAX_Y,
                                 config.MAX_X, config.OMS_PARAMS['threshold'])

        vSliceOMS[:] = OMS.squeeze(0)
        #window
        saliency_map[:], salmax_coords[:] = run_attention(
            window, net_attention, device, resolution, config.ATTENTION_PARAMS['num_pyr']
        )
        # #OMS
        # saliency_map[:], salmax_coords[:] = run_attention(
        #     vSliceOMS[:], net_attention, device, resolution, config.ATTENTION_PARAMS['num_pyr']
        # )

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
        # cv2.putText(window_img_color, 'Events', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

        mask = np.zeros_like(window_img_color)
        mask[y1:y2, x1:x2] = window_img_color[y1:y2, x1:x2]
        window_img_boxed = mask

        roi_center = coord_encoder.encode([[x, y]])

        if bbox_savingFLAG:
            name_file = data_file_i.split('.png')[0]
            save_path = os.path.join(bbox_saving_path+obj, f"{name_file}_bbox.png")
            save_img = np.zeros((box_size, box_size), dtype=np.uint8)
            crop = window_img[y1:y2, x1:x2]
            h, w = crop.shape[:2]
            save_img[:h, :w] = crop
            cv2.imwrite(save_path, save_img)

        # current_roi = clip_processor(images=[window_img_color[y1:y2, x1:x2]], return_tensors="pt", padding=True).to(device)
        # with torch.no_grad():
        #     image_features = clip_model.get_image_features(**current_roi)

        roi_crop = window_img_color[y1:y2, x1:x2]
        roi_tensor = event_transform(roi_crop).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = event_clip_model.clip.get_image_features(pixel_values=roi_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # L2-normalize
            image_features_np = image_features.cpu().numpy()  # ✅ numpy array of shape (1, 512)

        img_feat_ssp = sspspace.SSP(image_features_np)
        new_roi = roi_center * img_feat_ssp
        gamma = 0.99 # 0.99 for bbox
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
        f'/Users/giuliadangelo/Downloads/npc-av-learning/CRIB/workingmemory/{obj}_memory.npy',
        object_memory)
cv2.destroyAllWindows()
