import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import json
import os

# Import your model (make sure this path is correct)
from Code1Training import AutoencoderClassifier


class EmbeddingExtractor:
    def __init__(self, model_path="autoencoder-trained/model.pth",
                 info_path="autoencoder-trained/training_info.json"):
        """
        Initialize the embedding extractor

        Args:
            model_path: Path to the trained model weights
            info_path: Path to training info JSON file
        """
        self.device = torch.device("mps" if torch.backends.mps.is_available()
                                   else "cuda" if torch.cuda.is_available()
        else "cpu")

        print(f"Using device: {self.device}")

        # Load training info
        self.training_info = self.load_training_info(info_path)

        # Initialize and load model
        self.model = self.load_model(model_path)

        # Setup image preprocessing (same as training)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        print("✅ Embedding extractor ready!")

    def load_training_info(self, info_path):
        """Load training information"""
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                training_info = json.load(f)
            return training_info
        else:
            print("Warning: Training info not found, using default values")
            return {
                'embedding_dim': 512,
                'class_names': []
            }

    def load_model(self, model_path):
        """Load the trained autoencoder model"""
        try:
            # Initialize model with correct parameters
            model = AutoencoderClassifier(
                embedding_dim=self.training_info.get('embedding_dim', 512)
            ).to(self.device)

            # Load trained weights
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.eval()

            print(f"✅ Model loaded from: {model_path}")
            return model

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise e

    def preprocess_image(self, image_array):
        """
        Preprocess numpy array image for the model

        Args:
            image_array: numpy array of shape (H, W, 1) with values 0-255

        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        if not isinstance(image_array, np.ndarray):
            raise ValueError("Input must be a numpy array")

        # Convert to PIL Image for preprocessing
        # Handle different data types
        if image_array.dtype != np.uint8:
            # If values are in [0, 1] range, convert to [0, 255]
            if image_array.max() <= 1.0:
                image_array = (image_array * 255).astype(np.uint8)
            else:
                image_array = image_array.astype(np.uint8)

        image = Image.fromarray(image_array)

        # Apply preprocessing
        image_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension

        return image_tensor.to(self.device)

    def get_embeddings(self, image_array):
        """
        Extract embeddings from an image numpy array

        Args:
            image_array: numpy array of shape (H, W, 1) with values 0-255

        Returns:
            numpy.ndarray: Embedding vector of shape (embedding_dim,)
        """
        image_tensor = self.preprocess_image(image_array)

        with torch.no_grad():
            embeddings = self.model.get_embeddings(image_tensor)
            return embeddings.cpu().numpy().squeeze()

    def get_full_prediction(self, image_array):
        """
        Get embeddings, reconstruction, and classification prediction

        Args:
            image_array: numpy array of shape (H, W, 3) with values 0-255

        Returns:
            dict: Dictionary containing embeddings, reconstruction, and predictions
        """
        image_tensor = self.preprocess_image(image_array)

        with torch.no_grad():
            embeddings, reconstructed = self.model(image_tensor)

            # Convert reconstructed image back to numpy
            reconstructed_np = reconstructed.cpu().numpy().squeeze()
            # Denormalize the reconstructed image
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            reconstructed_np = reconstructed_np.transpose(1, 2, 0)  # CHW to HWC
            reconstructed_np = reconstructed_np * std + mean
            reconstructed_np = np.clip(reconstructed_np * 255, 0, 255).astype(np.uint8)

            return {
                'embeddings': embeddings.cpu().numpy().squeeze(),
                'reconstructed': reconstructed_np
            }

    def get_similarity(self, image_array1, image_array2):
        """
        Calculate cosine similarity between embeddings of two image arrays

        Args:
            image_array1, image_array2: numpy arrays of shape (H, W, 3)

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
