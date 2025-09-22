import os
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt


class AutoencoderClassifier(nn.Module):
    """Autoencoder model optimized for event frames (grayscale input)"""

    def __init__(self, embedding_dim=512, input_channels=1):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            # Block 1: 224x224 -> 112x112
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),

            # Block 2: 112x112 -> 56x56
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),

            # Block 3: 56x56 -> 28x28
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),

            # Block 4: 28x28 -> 7x7
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

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 512 * 7 * 7),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (512, 7, 7)),

            # 7x7 -> 14x14
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 14x14 -> 28x28
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 28x28 -> 56x56
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 56x56 -> 112x112
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 112x112 -> 224x224
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
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
        encoded = self.encoder(x)
        embeddings = self.embedding_layer(encoded)
        reconstructed = self.decoder(embeddings)
        return embeddings, reconstructed

    def get_embeddings(self, x):
        """Extract embeddings without reconstruction"""
        with torch.no_grad():
            encoded = self.encoder(x)
            embeddings = self.embedding_layer(encoded)
        return embeddings


class EventDataset(Dataset):
    """Dataset for event frame images"""

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
            image = Image.open(image_path).convert("L")  # Grayscale
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return self.__getitem__((idx + 1) % len(self.samples))


class EventFrameEmbeddingExtractor:
    """Embedding extractor for trained event frame autoencoder"""

    def __init__(self, model_path, info_path=None, device=None):
        # Set device
        if device is None:
            self.device = torch.device(
                "mps" if torch.backends.mps.is_available()
                else "cuda" if torch.cuda.is_available()
                else "cpu"
            )
        else:
            self.device = device

        # Load model configuration
        self.embedding_dim = 512
        self.input_channels = 1

        if info_path and os.path.exists(info_path):
            with open(info_path, 'r') as f:
                info = json.load(f)
                self.embedding_dim = info.get('embedding_dim', 512)
                self.input_channels = info.get('input_channels', 1)

        # Initialize and load model
        self.model = AutoencoderClassifier(
            embedding_dim=self.embedding_dim,
            input_channels=self.input_channels
        ).to(self.device)

        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        print(f"Model loaded successfully on {self.device}")

    def get_embeddings(self, image_array):
        """Extract embeddings from 2D grayscale image array"""
        with torch.no_grad():
            # Input validation
            if not isinstance(image_array, np.ndarray) or image_array.ndim != 2:
                raise ValueError("Expected 2D numpy array")

            # Resize if needed
            if image_array.shape != (224, 224):
                image_array = cv2.resize(image_array, (224, 224), interpolation=cv2.INTER_LINEAR)

            # Convert to tensor and normalize
            tensor = torch.from_numpy(image_array).float()
            tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
            tensor = (tensor / 255.0 - 0.5) / 0.5  # Normalize to [-1, 1]
            tensor = tensor.to(self.device)

            # Get embeddings
            embeddings = self.model.get_embeddings(tensor)
            return embeddings.cpu().numpy().flatten()


def get_transforms(train=True):
    """Get image transforms for training/validation"""
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])


def visualize_reconstructions(model, val_loader, device, save_path, num_samples=8):
    """Visualize original vs reconstructed images"""
    model.eval()

    # Get sample data
    data_iter = iter(val_loader)
    images, _ = next(data_iter)
    images = images[:num_samples].to(device)

    with torch.no_grad():
        _, reconstructed = model(images)

    # Denormalize
    mean = torch.tensor([0.5]).view(1, 1, 1, 1).to(device)
    std = torch.tensor([0.5]).view(1, 1, 1, 1).to(device)

    original = torch.clamp(images * std + mean, 0, 1)
    reconstructed = torch.clamp(reconstructed * std + mean, 0, 1)

    # Create visualization
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))

    for i in range(num_samples):
        orig_img = original[i].cpu().squeeze().numpy()
        axes[0, i].imshow(orig_img, cmap='gray')
        axes[0, i].set_title('Original')
        axes[0, i].axis('off')

        recon_img = reconstructed[i].cpu().squeeze().numpy()
        axes[1, i].imshow(recon_img, cmap='gray')
        axes[1, i].set_title('Reconstructed')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def train_model(data_root, results_dir, config=None):
    """Train the autoencoder model"""
    # Default configuration
    default_config = {
        'batch_size': 32,
        'num_epochs': 50,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'embedding_dim': 512,
        'patience': 10
    }

    if config:
        default_config.update(config)

    cfg = default_config

    # Setup device
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")
    os.makedirs(results_dir, exist_ok=True)

    # Get class names
    class_names = sorted([d for d in os.listdir(data_root)
                          if os.path.isdir(os.path.join(data_root, d)) and not d.startswith('.')])
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    print(f"Found {len(class_names)} classes: {class_names}")

    # Create datasets
    full_dataset = EventDataset(data_root, class_to_idx, transform=get_transforms(train=True))

    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=4)

    # Initialize model
    model = AutoencoderClassifier(embedding_dim=cfg['embedding_dim']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    criterion = nn.MSELoss()

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(cfg['num_epochs']):
        # Training phase
        model.train()
        train_loss = 0.0

        for images, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg['num_epochs']}"):
            images = images.to(device)

            optimizer.zero_grad()
            _, reconstructed = model(images)
            loss = criterion(reconstructed, images)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                _, reconstructed = model(images)
                loss = criterion(reconstructed, images)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        scheduler.step()

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'embedding_dim': cfg['embedding_dim'],
                'val_loss': val_loss
            }, os.path.join(results_dir, "best_model.pth"))

            # Visualize reconstructions
            visualize_reconstructions(model, val_loader, device,
                                      os.path.join(results_dir, 'reconstructions.png'))
            print(f"  ✅ New best model saved! Val Loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= cfg['patience']:
                print(f"Early stopping after {cfg['patience']} epochs without improvement")
                break

    # Save final artifacts
    model_dir = os.path.join(results_dir, "final_model")
    os.makedirs(model_dir, exist_ok=True)

    # Load best model
    checkpoint = torch.load(os.path.join(results_dir, "best_model.pth"))
    model.load_state_dict(checkpoint['model_state_dict'])
    torch.save(model.state_dict(), os.path.join(model_dir, "model.pth"))

    # Save training info
    training_info = {
        **cfg,
        'input_channels': 1,
        'data_type': 'event_frames_grayscale',
        'best_val_loss': best_val_loss,
        'class_names': class_names
    }

    with open(os.path.join(model_dir, "training_info.json"), "w") as f:
        json.dump(training_info, f, indent=2)

    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(model_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ Training complete! Model saved to: {model_dir}")
    return model, training_info


# Example usage
if __name__ == "__main__":
    # Update these paths to your actual data locations
    DATA_ROOT = '/media/matt/bigdata/DATA/CRIB/bboxes/'
    RESULTS_DIR = '/media/matt/bigdata/DATA/CRIB/resultsbbox30050epochs/'

    # Custom configuration (optional)
    config = {
        'num_epochs': 50,
        'batch_size': 32,
        'lr': 1e-3,
        'embedding_dim': 512
    }

    # Train model
    model, info = train_model(DATA_ROOT, RESULTS_DIR, config)

    # Use embedding extractor
    extractor = EventFrameEmbeddingExtractor(
        model_path=os.path.join(RESULTS_DIR, "final_model/model.pth"),
        info_path=os.path.join(RESULTS_DIR, "final_model/training_info.json")
    )

    # Extract embeddings from a sample image
    sample_image = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
    embeddings = extractor.get_embeddings(sample_image)
    print(f"Embeddings shape: {embeddings.shape}")