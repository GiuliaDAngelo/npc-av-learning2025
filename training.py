import os
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F

# --------------------------
# CONFIG
# --------------------------
device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available()
                      else "cpu")

data_root = "/Users/giuliadangelo/Downloads/npc-av-learning/CRIB/train_data/bbox/"
batch_size = 16
num_epochs = 10
lr = 5e-6  # Small learning rate for fine-tuning
model_name = "openai/clip-vit-base-patch32"

# --------------------------
# PREPROCESSOR
# --------------------------
processor = CLIPProcessor.from_pretrained(model_name)

# --------------------------
# TEXT LABELS
# --------------------------
class_names = sorted(os.listdir(data_root))
text_prompts = [f"a photo of a {label}" for label in class_names]

# --------------------------
# CUSTOM DATASET: (image, matching text)
# --------------------------
class CLIPEventDataset(Dataset):
    def __init__(self, root_dir, processor, class_names):
        self.samples = []
        self.processor = processor
        self.class_names = class_names
        for label in class_names:
            label_path = os.path.join(root_dir, label)
            # skip hidden folders starting with '.'
            if not os.path.isdir(label_path) or label.startswith('.'):
                continue
            for fname in os.listdir(label_path):
                if fname.endswith(".png"):
                    self.samples.append((os.path.join(label_path, fname), f"a photo of a {label}"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, text = self.samples[idx]
        image = Image.open(image_path).convert("RGB")  # CLIP expects RGB
        return image, text

dataset = CLIPEventDataset(data_root, processor, class_names)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --------------------------
# MODEL
# --------------------------
model = CLIPModel.from_pretrained(model_name).to(device)
model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# --------------------------
# TRAINING LOOP (Contrastive loss)
# --------------------------
for epoch in range(num_epochs):
    total_loss = 0.0
    correct = 0
    total = 0
    loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for batch in loop:
        images, texts = batch
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(device)

        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # [batch_size, batch_size]
        logits_per_text = outputs.logits_per_text

        labels = torch.arange(len(images)).to(device)  # ground truth: i-th image matches i-th text

        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        loss = (loss_i + loss_t) / 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits_per_image, dim=1)
        correct += (preds == labels).sum().item()
        total += len(labels)
        total_loss += loss.item()

        loop.set_postfix(loss=loss.item(), acc=100 * correct / total)

    print(f"Epoch {epoch+1}: Loss={total_loss:.4f}, Accuracy={100 * correct / total:.2f}%")

# --------------------------
# SAVE MODEL + PROCESSOR
# --------------------------
model.save_pretrained("clip-event-finetuned")
processor.save_pretrained("clip-event-finetuned")
print("✅ Model and processor saved to 'clip-event-finetuned'")
