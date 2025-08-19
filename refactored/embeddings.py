import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch

import torchvision.transforms as T
from transformers import CLIPModel, CLIPProcessor
from PIL import Image


def get_image_patch(image, fixation, size):
    x,y = fixation
    half_size = size // 2
    return image[y-half_size:y+half_size, x-half_size:x+half_size]


class CLIPEmbeddings:
    def __init__(self, model_name='openai/clip-vit-base-patch32', device='cuda'):
        self.device = device
        self.clip_model = CLIPModel.from_pretrained(model_name).to(device)
        self.clip_processor = CLIPProcessor.from_pretrained(model_name)

    def get_embedding(self, image):
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy().squeeze()

    def get_embeddings(self, images):
        try:
            inputs = self.clip_processor(images=images, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            return image_features.cpu().numpy()
        except Exception as e:
            print(f"Error in get_embeddings: {e}")
            return np.array([])

