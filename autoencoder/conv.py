# https://github.com/avigupta2612/Autoencoders/tree/master/

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import glob

device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration parameters
IMG_SIZE = 100  # Adjustable image size parameter
BATCH_SIZE = 20

# Early stopping parameters
EARLY_STOPPING_PATIENCE = 10  # Number of epochs to wait for improvement
EARLY_STOPPING_MIN_DELTA = 1e-4  # Minimum change to qualify as improvement

class PatchesDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = glob.glob(os.path.join(data_dir, "*.png"))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale

        if self.transform:
            image = self.transform(image)

        return image, image  # Return image twice for autoencoder (input, target)

# Define transforms for grayscale images
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# Load training data and split into train/val (80/20)
train_dataset = PatchesDataset("patches_data/train", transform=transform)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_data, val_data = random_split(train_dataset, [train_size, val_size])

# Load test data
test_data = PatchesDataset("patches_data/test", transform=transform)

print(f"Training samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")
print(f"Test samples: {len(test_data)}")

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)


class ConvAutoencoder(nn.Module):
    def __init__(self, img_size=IMG_SIZE):
        super(ConvAutoencoder, self).__init__()
        self.img_size = img_size

        # Encoder
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Calculate the size after pooling layers
        # After 2 pooling layers: img_size -> img_size//2 -> img_size//4
        self.encoded_size = img_size // 4

        # Decoder
        self.tconv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.tconv2 = nn.ConvTranspose2d(16, 1, 2, stride=2)

    def forward(self, x):
        # Encoder
        x = self.pool(F.relu(self.conv1(x)))  # img_size -> img_size//2
        x = self.pool(F.relu(self.conv2(x)))  # img_size//2 -> img_size//4

        # Decoder
        x = F.relu(self.tconv1(x))  # img_size//4 -> img_size//2
        x = F.sigmoid(self.tconv2(x))  # img_size//2 -> img_size

        # Ensure output matches input size exactly
        if x.size(-1) != self.img_size or x.size(-2) != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)

        return x
    
model = ConvAutoencoder()
print(model)
# put model on device
model = model.to(device)

# Create checkpoint directory
os.makedirs("patches_conv", exist_ok=True)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 20
best_val_loss = float('inf')

# Lists to store loss history for plotting
train_losses = []
val_losses = []

# Early stopping variables
epochs_without_improvement = 0
early_stopped = False

for epoch in range(epochs):
    # Training phase
    model.train()
    train_loss = 0.0

    for images, targets in train_loader:
        # put images on device
        images = images.to(device)
        targets = targets.to(device)

        # clear the gradients
        optimizer.zero_grad()
        # forward pass, backward pass, optimize
        output = model(images)
        # compute the loss
        loss = criterion(output, targets)
        # backward pass
        loss.backward()
        # update the weights
        optimizer.step()
        train_loss += loss.item() * images.size(0)

    train_loss = train_loss / len(train_data)

    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)
            output = model(images)
            loss = criterion(output, targets)
            val_loss += loss.item() * images.size(0)

    val_loss = val_loss / len(val_data)

    # Store losses for plotting
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch: {epoch + 1}/{epochs} \tTrain Loss: {train_loss:.6f} \tVal Loss: {val_loss:.6f}")

    # Check for improvement in validation loss
    if val_loss < best_val_loss - EARLY_STOPPING_MIN_DELTA:
        # Significant improvement found
        best_val_loss = val_loss
        epochs_without_improvement = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
        }, 'patches_conv/best_model.ckpt')
        print(f"New best model saved with validation loss: {val_loss:.6f}")
    else:
        # No significant improvement
        epochs_without_improvement += 1
        print(f"No improvement for {epochs_without_improvement} epoch(s)")

    # Early stopping check
    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
        print(f"Early stopping triggered! No improvement for {EARLY_STOPPING_PATIENCE} epochs.")
        print(f"Best validation loss: {best_val_loss:.6f}")
        early_stopped = True
        break

    # Save checkpoint every 5 epochs
    if (epoch + 1) % 5 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
        }, f'patches_conv/checkpoint_epoch_{epoch+1}.ckpt')

# Save final model (only if not early stopped, since best model is already saved)
if not early_stopped:
    torch.save({
        'epoch': epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss
    }, 'patches_conv/final_model.ckpt')

if early_stopped:
    print(f"Training stopped early after {len(train_losses)} epochs. Best model saved in patches_conv/ directory.")
else:
    print("Training completed. Models saved in patches_conv/ directory.")

def save_reconstruction_plots(model, data_loader, dataset_name, num_samples=10):
    """Save reconstruction plots for a given dataset"""
    model.eval()

    # Get a batch of images
    dataiter = iter(data_loader)
    images, targets = next(dataiter)
    images = images.to(device)

    # Get reconstructions
    with torch.no_grad():
        reconstructions = model(images)

    # Move to CPU for plotting
    images = images.cpu().numpy()
    reconstructions = reconstructions.cpu().numpy()

    # Create the plot
    fig, axes = plt.subplots(nrows=2, ncols=num_samples, figsize=(20, 4))
    fig.suptitle(f'{dataset_name} Images vs Reconstructions', fontsize=16)

    for i in range(num_samples):
        # Original images on top row
        axes[0, i].imshow(np.squeeze(images[i]), cmap='gray')
        axes[0, i].set_title('Original' if i == 0 else '')
        axes[0, i].axis('off')

        # Reconstructions on bottom row
        axes[1, i].imshow(np.squeeze(reconstructions[i]), cmap='gray')
        axes[1, i].set_title('Reconstruction' if i == 0 else '')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(f'patches_conv/{dataset_name.lower()}_reconstructions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {dataset_name.lower()}_reconstructions.png")

# Save reconstruction plots for train data
save_reconstruction_plots(model, train_loader, "Train")

# Save reconstruction plots for test data
save_reconstruction_plots(model, test_loader, "Test")

# Plot and save training/validation loss curves
plt.figure(figsize=(10, 6))
actual_epochs = len(train_losses)
plt.plot(range(1, actual_epochs + 1), train_losses, 'b-', label='Training Loss', linewidth=2)
plt.plot(range(1, actual_epochs + 1), val_losses, 'r-', label='Validation Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Time')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('patches_conv/loss_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved loss_curves.png")

# Test the model on a batch of test images (keep original visualization)
dataiter = iter(test_loader)
images, targets = next(dataiter)

# put images on device
images = images.to(device)

# put the model in evaluation mode
model.eval()

# get sample outputs
with torch.no_grad():
    output = model(images)

# put images and output on cpu
images = images.cpu().numpy()
# output is resized into a batch of images
output = output.view(BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE)
# use detach when it's an output that requires_grad
output = output.cpu().detach().numpy()

# plot the first ten input images and then reconstructed images
fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25,4))

# input images on top row, reconstructions on bottom
for img_batch, row in zip([images, output], axes):
    for img, ax in zip(img_batch, row):
        ax.imshow(np.squeeze(img), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
plt.show()
