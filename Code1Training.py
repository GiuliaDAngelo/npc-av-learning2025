import os
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
import json
import matplotlib.pyplot as plt
import numpy as np

# --------------------------
# CONFIG
# --------------------------
device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available()
else "cpu")

data_root = "/Users/giuliadangelo/workspace/data/DATASETs/CRIB/CRIB400/train_data/bboxes/"
results_dir = "/Users/giuliadangelo/workspace/data/DATASETs/CRIB/CRIB400/train_data/resultsbbox30050epochs/"

batch_size = 32
num_epochs = 50 #we already tried with 8
lr = 1e-3 #1e-3
weight_decay = 1e-4
# New parameters for autoencoder
embedding_dim = 512
reconstruction_weight = 0.5  # Weight for reconstruction loss
classification_weight = 0.5  # Weight for classification loss


# --------------------------
# AUTOENCODER + CLASSIFIER MODEL (UPDATED FOR EVENT FRAMES)
# --------------------------
class AutoencoderClassifier(nn.Module):
    def __init__(self, num_classes, embedding_dim=512, input_channels=1):  # CHANGED: default to 1 channel
        super().__init__()

        # Shared encoder - optimized for event frames
        self.encoder = nn.Sequential(
            # First conv block
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),

            # Second conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),

            # Third conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),

            # Fourth conv block
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7)),
        )

        # Embedding layer
        self.embedding_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

        # Decoder for reconstruction (UPDATED FOR EVENT FRAMES)
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 512 * 7 * 7),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (512, 7, 7)),

            # First upsampling: 7x7 -> 14x14
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Second upsampling: 14x14 -> 28x28
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Third upsampling: 28x28 -> 56x56
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Fourth upsampling: 56x56 -> 112x112
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Fifth upsampling: 112x112 -> 224x224 (CHANGED: output 1 channel for grayscale)
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Output between 0 and 1
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encode
        encoded = self.encoder(x)
        embeddings = self.embedding_layer(encoded)

        # Decode for reconstruction
        reconstructed = self.decoder(embeddings)

        # Classify
        class_logits = self.classifier(embeddings)

        return embeddings, reconstructed, class_logits

    def get_embeddings(self, x):
        """Extract embeddings without reconstruction/classification"""
        with torch.no_grad():
            encoded = self.encoder(x)
            embeddings = self.embedding_layer(encoded)
        return embeddings


# --------------------------
# DATASET (UPDATED FOR EVENT FRAMES)
# --------------------------
class EventDataset(Dataset):
    def __init__(self, root_dir, class_to_idx, transform=None):
        self.samples = []
        self.class_to_idx = class_to_idx
        self.transform = transform

        for class_name in class_to_idx.keys():
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            for fname in os.listdir(class_path):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(class_path, fname), class_to_idx[class_name]))

        print(f"Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        try:
            # CHANGED: Convert to grayscale for event frames
            image = Image.open(image_path).convert("L")  # Force grayscale for event frames

            if self.transform:
                image = self.transform(image)

            return image, label
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return self.__getitem__((idx + 1) % len(self.samples))


# --------------------------
# TRANSFORMS (UPDATED FOR EVENT FRAMES)
# --------------------------
def get_event_transforms(train=True):
    """Optimized transforms for event frames"""
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            # Event frame friendly augmentations
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),  # Reduced for event frames
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),  # Gentle transforms
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # CHANGED: Single channel normalization
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # CHANGED: Single channel normalization
        ])


# --------------------------
# VISUALIZATION HELPER (UPDATED FOR GRAYSCALE)
# --------------------------
def visualize_reconstructions(model, val_loader, device, num_samples=8):
    """Visualize original vs reconstructed event frames"""
    model.eval()

    # Get a batch of validation data
    data_iter = iter(val_loader)
    images, labels = next(data_iter)
    images = images[:num_samples].to(device)

    with torch.no_grad():
        embeddings, reconstructed, _ = model(images)

    # CHANGED: Denormalize for grayscale visualization
    mean = torch.tensor([0.5]).view(1, 1, 1, 1).to(device)
    std = torch.tensor([0.5]).view(1, 1, 1, 1).to(device)

    original_denorm = images * std + mean
    reconstructed_denorm = reconstructed * std + mean

    # Clamp to [0, 1]
    original_denorm = torch.clamp(original_denorm, 0, 1)
    reconstructed_denorm = torch.clamp(reconstructed_denorm, 0, 1)

    # Create visualization
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))

    for i in range(num_samples):
        # CHANGED: Handle grayscale images properly
        orig_img = original_denorm[i].cpu().squeeze().numpy()  # Remove channel dimension
        axes[0, i].imshow(orig_img, cmap='gray')  # Use gray colormap
        axes[0, i].set_title('Original Event Frame')
        axes[0, i].axis('off')

        # Reconstructed event frame
        recon_img = reconstructed_denorm[i].cpu().squeeze().numpy()  # Remove channel dimension
        axes[1, i].imshow(recon_img, cmap='gray')  # Use gray colormap
        axes[1, i].set_title('Reconstructed')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(results_dir+'reconstruction_samples.png', dpi=150, bbox_inches='tight')
    plt.close()


# --------------------------
# TRAINING FUNCTION (UPDATED FOR EVENT FRAMES)
# --------------------------
def train_model():
    print(f"Using device: {device}")
    print("🎯 Configuring for EVENT FRAMES (grayscale)")

    # CHANGED: Event frames are always grayscale - set to 1 channel
    input_channels = 1
    print(f"✓ Using {input_channels} input channel for event frames")

    # Get class names
    class_names = sorted([d for d in os.listdir(data_root)
                          if os.path.isdir(os.path.join(data_root, d)) and not d.startswith('.')])
    print(f"Found {len(class_names)} classes: {class_names}")

    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

    # Create datasets
    train_dataset = EventDataset(data_root, class_to_idx, transform=get_event_transforms(train=True))
    val_dataset = EventDataset(data_root, class_to_idx, transform=get_event_transforms(train=False))

    # Split dataset
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, _ = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    val_dataset, _ = torch.utils.data.random_split(val_dataset, [val_size, len(val_dataset) - val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # CHANGED: Initialize model with 1 input channel for event frames
    model = AutoencoderClassifier(len(class_names),
                                 embedding_dim=embedding_dim,
                                 input_channels=input_channels).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizers and schedulers
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    # Loss functions
    classification_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    reconstruction_criterion = nn.MSELoss()

    # Training loop
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_recon_loss = 0.0
        train_class_loss = 0.0
        train_correct = 0
        train_total = 0

        train_loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")

        for batch_idx, (images, labels) in enumerate(train_loop):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            embeddings, reconstructed, class_logits = model(images)

            # Calculate losses
            recon_loss = reconstruction_criterion(reconstructed, images)
            class_loss = classification_criterion(class_logits, labels)

            # Combined loss
            total_loss = reconstruction_weight * recon_loss + classification_weight * class_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Statistics
            _, predicted = torch.max(class_logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_loss += total_loss.item()
            train_recon_loss += recon_loss.item()
            train_class_loss += class_loss.item()

            train_loop.set_postfix(
                total_loss=total_loss.item(),
                recon_loss=recon_loss.item(),
                class_loss=class_loss.item(),
                acc=100 * train_correct / train_total
            )

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_recon_loss = 0.0
        val_class_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]"):
                images, labels = images.to(device), labels.to(device)

                embeddings, reconstructed, class_logits = model(images)

                recon_loss = reconstruction_criterion(reconstructed, images)
                class_loss = classification_criterion(class_logits, labels)
                total_loss = reconstruction_weight * recon_loss + classification_weight * class_loss

                _, predicted = torch.max(class_logits.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_loss += total_loss.item()
                val_recon_loss += recon_loss.item()
                val_class_loss += class_loss.item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        scheduler.step()

        # Calculate accuracies
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total

        # Store metrics
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(
            f"  Train - Total: {train_loss / len(train_loader):.4f}, Recon: {train_recon_loss / len(train_loader):.4f}, Class: {train_class_loss / len(train_loader):.4f}, Acc: {train_acc:.2f}%")
        print(
            f"  Val - Total: {val_loss / len(val_loader):.4f}, Recon: {val_recon_loss / len(val_loader):.4f}, Class: {val_class_loss / len(val_loader):.4f}, Acc: {val_acc:.2f}%")

        # Early stopping and model saving
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': class_names,
                'class_to_idx': class_to_idx,
                'embedding_dim': embedding_dim,
                'reconstruction_weight': reconstruction_weight,
                'classification_weight': classification_weight,
                'input_channels': input_channels,  # ADDED: Save input channels info
            }, os.path.join(results_dir, "best_autoencoder_classifier.pth"))
            print(f"  ✅ New best model saved! Val Acc: {val_acc:.2f}%")

            # Save reconstruction visualizations
            visualize_reconstructions(model, val_loader, device)

        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement")
                break

    # Final evaluation
    checkpoint = torch.load(os.path.join(results_dir, "best_autoencoder_classifier.pth"))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Final predictions
    all_preds = []
    all_labels = []
    all_embeddings = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Final Evaluation"):
            images = images.to(device)
            embeddings, _, class_logits = model(images)
            _, predicted = torch.max(class_logits.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_embeddings.extend(embeddings.cpu().numpy())

    final_acc = accuracy_score(all_labels, all_preds)
    print(f"\nFinal Validation Accuracy: {final_acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Save final model and embeddings
    os.makedirs(results_dir+"autoencoder-classifier-trained", exist_ok=True)
    torch.save(model.state_dict(), results_dir+"autoencoder-classifier-trained/model.pth")

    # Save embeddings
    np.save(results_dir+"autoencoder-classifier-trained/embeddings.npy", np.array(all_embeddings))
    np.save(results_dir+"autoencoder-classifier-trained/labels.npy", np.array(all_labels))

    # Save training info with updated metadata
    with open(os.path.join(results_dir, "autoencoder-classifier-trained/training_info.json"), "w") as f:
        json.dump({
            "class_names": class_names,
            "class_to_idx": class_to_idx,
            "final_accuracy": final_acc,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "embedding_dim": embedding_dim,
            "reconstruction_weight": reconstruction_weight,
            "classification_weight": classification_weight,
            "num_classes": len(class_names),
            "input_channels": input_channels,  # ADDED
            "data_type": "event_frames_grayscale"  # ADDED
        }, f, indent=2)

    # Plot training curves
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss (Event Frames)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.title('Training and Validation Accuracy (Event Frames)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(results_dir+'autoencoder-classifier-trained/training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("✅ Event frame model, embeddings, and training info saved to 'autoencoder-classifier-trained'")


if __name__ == '__main__':
    train_model()