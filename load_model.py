import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import os

# ConvAutoencoder class (same as in conv.py)
class ConvAutoencoder(nn.Module):
    def __init__(self, img_size=100):
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
        encoded = self.pool(F.relu(self.conv2(x)))  # img_size//2 -> img_size//4

        # Decoder
        x = F.relu(self.tconv1(encoded))  # img_size//4 -> img_size//2
        x = F.sigmoid(self.tconv2(x))  # img_size//2 -> img_size

        # Ensure output matches input size exactly
        if x.size(-1) != self.img_size or x.size(-2) != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)

        return x

    def get_embeddings(self, x):
        """Extract embeddings (encoded representation) from input"""
        # Encoder only
        x = self.pool(F.relu(self.conv1(x)))  # img_size -> img_size//2
        encoded = self.pool(F.relu(self.conv2(x)))  # img_size//2 -> img_size//4
        encoded = encoded.view(encoded.size(0), -1)
        # Normalize the encoded representation
        #print('ENCODED SHAPE', encoded.shape)
        encoded = encoded / (torch.norm(encoded, dim=1, keepdim=True) + 1e-8)
        # set nans to zero
        # TODO: retrain the autoencoder with normalized embeddings
        #encoded[torch.isnan(encoded)] = 0
        # Flatten the encoded representation
        return encoded


class EmbeddingExtractor:
    def __init__(self, model_path="autoencoder/patches_conv/best_model.ckpt", img_size=100):
        """
        Initialize the embedding extractor

        Args:
            model_path: Path to the trained model checkpoint
            img_size: Image size used during training (default: 100)
        """
        self.device = torch.device("mps" if torch.backends.mps.is_available()
                                   else "cuda" if torch.cuda.is_available()
                                   else "cpu")
        self.img_size = img_size

        print(f"Using device: {self.device}")

        # Initialize and load model
        self.model = self.load_model(model_path)

        # Setup image preprocessing (same as training)
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()
        ])

        print("✅ Embedding extractor ready!")

    def load_model(self, model_path):
        """Load the trained conv autoencoder model"""
        try:
            # Initialize model
            model = ConvAutoencoder(img_size=self.img_size).to(self.device)

            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            print(f"✅ Model loaded from: {model_path}")
            print(f"   Epoch: {checkpoint['epoch']}")
            print(f"   Train Loss: {checkpoint['train_loss']:.6f}")
            print(f"   Val Loss: {checkpoint['val_loss']:.6f}")
            return model

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise e

    def preprocess_image(self, image_array):
        """
        Preprocess numpy array image for the model

        Args:
            image_array: numpy array of shape (H, W) or (H, W, 1) with values 0-255

        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        if not isinstance(image_array, np.ndarray):
            raise ValueError("Input must be a numpy array")

        # Handle different input shapes
        if len(image_array.shape) == 3 and image_array.shape[2] == 1:
            # Remove single channel dimension
            image_array = image_array.squeeze(axis=2)
        elif len(image_array.shape) == 3:
            # Convert RGB to grayscale if needed
            if image_array.shape[2] == 3:
                image_array = np.dot(image_array[...,:3], [0.2989, 0.5870, 0.1140])

        # Handle different data types
        if image_array.dtype != np.uint8:
            # If values are in [0, 1] range, convert to [0, 255]
            if image_array.max() <= 1.0:
                image_array = (image_array * 255).astype(np.uint8)
            else:
                image_array = image_array.astype(np.uint8)

        # Convert to PIL Image for preprocessing (mode 'L' for grayscale)
        image = Image.fromarray(image_array, mode='L')

        # Apply preprocessing
        image_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension

        return image_tensor.to(self.device)

    def get_embeddings(self, image_array):
        """
        Extract embeddings from an image numpy array

        Args:
            image_array: numpy array of shape (H, W) or (H, W, 1) with values 0-255

        Returns:
            numpy.ndarray: Embedding vector (flattened encoded representation)
        """
        image_tensor = self.preprocess_image(image_array)

        with torch.no_grad():
            embeddings = self.model.get_embeddings(image_tensor)
            return embeddings.cpu().numpy().squeeze()

    def get_reconstruction(self, image_array):
        """
        Get reconstruction of the input image

        Args:
            image_array: numpy array of shape (H, W) or (H, W, 1) with values 0-255

        Returns:
            numpy.ndarray: Reconstructed image as numpy array
        """
        image_tensor = self.preprocess_image(image_array)

        with torch.no_grad():
            reconstructed = self.model(image_tensor)
            # Convert back to numpy and remove batch dimension
            reconstructed_np = reconstructed.cpu().numpy().squeeze()
            # Convert from [0,1] to [0,255]
            reconstructed_np = np.clip(reconstructed_np * 255, 0, 255).astype(np.uint8)
            return reconstructed_np

    def get_full_prediction(self, image_array):
        """
        Get both embeddings and reconstruction

        Args:
            image_array: numpy array of shape (H, W) or (H, W, 1) with values 0-255

        Returns:
            dict: Dictionary containing embeddings and reconstruction
        """
        return {
            'embeddings': self.get_embeddings(image_array),
            'reconstructed': self.get_reconstruction(image_array)
        }

    def get_similarity(self, image_array1, image_array2):
        """
        Calculate cosine similarity between embeddings of two image arrays

        Args:
            image_array1, image_array2: numpy arrays of shape (H, W) or (H, W, 1)

        Returns:
            float: Cosine similarity score
        """
        emb1 = self.get_embeddings(image_array1)
        emb2 = self.get_embeddings(image_array2)

        # Normalize embeddings
        emb1_norm = emb1 / np.linalg.norm(emb1)
        emb2_norm = emb2 / np.linalg.norm(emb2)

        # Calculate cosine similarity
        similarity = np.dot(emb1_norm, emb2_norm)
        return similarity


# Example usage
if __name__ == "__main__":
    # Initialize the extractor
    extractor = EmbeddingExtractor()

    # Example with a random numpy array (replace with your actual image data)
    # For testing purposes - create a random 100x100 grayscale image
    test_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

    # Get embeddings
    embeddings = extractor.get_embeddings(test_image)
    print(f"Embeddings shape: {embeddings.shape}")

    # Get reconstruction
    reconstruction = extractor.get_reconstruction(test_image)
    print(f"Reconstruction shape: {reconstruction.shape}")

    # Get both
    full_pred = extractor.get_full_prediction(test_image)
    print(f"Full prediction keys: {full_pred.keys()}")
