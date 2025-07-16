import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import numpy as np
import os
import torchvision.transforms as transforms
from training import CLIPEventClassifier  # Make sure this class is defined and matches your trained model

# ======= CONFIG =======
device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available()
                      else "cpu")

bbox_dir = "/Users/giuliadangelo/Downloads/npc-av-learning/CRIB/train_data/bbox/"
model_path = "clip_event_classifier.pth"

# ======= TEXT LABELS =======
text_labels = [
    "bow", "baby", "balls", "basketball", "bee", "bike_helmet", "birdie",
    "bulb", "bunny", "comb", "cookie", "dog", "dolphin", "doraemon",
    "film_clapper", "fork", "fox"
]
text_labels = np.array(text_labels).flatten()
text_labels_list = ["a photo of a " + t for t in text_labels]

# ======= Load CLIP text embeddings =======
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

text_tokens = clip_processor(text=text_labels_list, return_tensors="pt", padding=True).to(device)
with torch.no_grad():
    text_embeddings = clip_model.get_text_features(**text_tokens)
    text_embeddings = text_embeddings / text_embeddings.norm(p=2, dim=-1, keepdim=True)

# ======= Load your fine-tuned model (for image embedding) =======
event_clip_model = CLIPEventClassifier(num_classes=len(text_labels)).to(device)
event_clip_model.load_state_dict(torch.load(model_path, map_location=device))
event_clip_model.eval()

# Use just the image encoder (without classifier)
def get_image_features(model, image_tensor):
    with torch.no_grad():
        img_feat = model.clip.get_image_features(pixel_values=image_tensor)
        return img_feat / img_feat.norm(dim=-1, keepdim=True)

# ======= Preprocessing (same as training) =======
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ======= Evaluation =======
acc = 0
total = 0

for label in text_labels:
    label_dir = os.path.join(bbox_dir, label)
    if not os.path.exists(label_dir):
        print(f"Missing directory for label: {label}")
        continue

    crop_files = [f for f in os.listdir(label_dir) if f.endswith(".png")]
    if len(crop_files) == 0:
        print(f"No crops for label: {label}")
        continue

    embeddings = []

    for crop_file in crop_files:
        crop_path = os.path.join(label_dir, crop_file)
        image = Image.open(crop_path).convert("L")
        image_tensor = transform(np.array(image)).unsqueeze(0).to(device)

        embedding = get_image_features(event_clip_model, image_tensor)
        embeddings.append(embedding.squeeze(0))  # shape: [512]

    if len(embeddings) == 0:
        print(f"No valid embeddings for label: {label}")
        continue

    # Average all embeddings
    avg_embedding = torch.stack(embeddings).mean(dim=0)
    avg_embedding = avg_embedding / avg_embedding.norm()
    avg_embedding = avg_embedding.unsqueeze(1)  # shape: [512, 1]

    # Compute similarity between avg_embedding and all text embeddings
    scores = torch.matmul(text_embeddings, avg_embedding).squeeze()  # shape: [num_labels]
    pred_idx = torch.argmax(scores).item()
    pred = text_labels[pred_idx]

    print(f"Label: {label} → Predicted: {pred}")
    if pred == label:
        acc += 1
    total += 1

# ======= Final accuracy =======
if total > 0:
    print(f"\nAccuracy: {acc / total:.3f} ({acc}/{total})")
else:
    print("No valid crops found for any label.")
