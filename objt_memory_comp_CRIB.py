import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import numpy as np
import os
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
import json
from collections import defaultdict
import pandas as pd

from load_model import EmbeddingExtractor

# ======= CONFIG =======
device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available()
else "cpu")

bbox_dir = "/Users/giuliadangelo/Downloads/npc-av-learning/CRIB/train_data/bbox/"
model_path = "event-cnn-trained/model.pth"



# ======= DISCOVER ALL OBJECTS =======
def discover_all_objects(bbox_dir):
    """Discover all object directories in the bbox folder"""
    objects = []
    if os.path.exists(bbox_dir):
        for item in os.listdir(bbox_dir):
            item_path = os.path.join(bbox_dir, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                # Check if directory has any image files
                image_files = [f for f in os.listdir(item_path)
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if image_files:
                    objects.append(item)
    return sorted(objects)


# Get all objects from your data
all_objects = discover_all_objects(bbox_dir)
print(f"Found {len(all_objects)} objects: {all_objects}")

# Use discovered objects as labels
text_labels = np.array(all_objects)
text_labels_list = ["a photo of a " + t for t in text_labels]
num_classes = len(text_labels)

# ======= Load CLIP for text embeddings =======
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

`extractor` = EmbeddingExtractor()


print("Computing CLIP text embeddings...")
text_tokens = clip_processor(text=text_labels_list, return_tensors="pt", padding=True).to(device)
with torch.no_grad():
    text_embeddings = clip_model.get_text_features(**text_tokens)
    text_embeddings = text_embeddings / text_embeddings.norm(p=2, dim=-1, keepdim=True)


# ======= Feature extraction from CNN =======
def get_image_features_cnn(model, image_tensor):
    """Extract features from CNN model"""
    with torch.no_grad():
        # Get features before the final classification layer
        features = model.features(image_tensor)
        # Global average pooling
        features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
        features = features.view(features.size(0), -1)
        # Normalize features
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        return features


# ======= Preprocessing (matching your training) =======
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ======= Data Analysis =======
print("\n" + "=" * 50)
print("STARTING COMPREHENSIVE DATA ANALYSIS")
print("=" * 50)

# Statistics collection
object_stats = defaultdict(dict)
all_embeddings = []
all_labels = []
similarity_matrix = torch.zeros((num_classes, num_classes), dtype=torch.float32)

acc = 0
total = 0

# Analyze each object
for i, label in enumerate(text_labels):
    print(f"\\nAnalyzing object {i + 1}/{len(text_labels)}: {label}")

    label_dir = os.path.join(bbox_dir, label)
    if not os.path.exists(label_dir):
        print(f"  ❌ Missing directory for label: {label}")
        object_stats[label]['status'] = 'missing_directory'
        continue

    crop_files = [f for f in os.listdir(label_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if len(crop_files) == 0:
        print(f"  ❌ No image files for label: {label}")
        object_stats[label]['status'] = 'no_images'
        continue

    print(f"  Found {len(crop_files)} image files")
    object_stats[label]['num_images'] = len(crop_files)

    embeddings = []
    valid_files = 0

    for crop_file in crop_files:
        crop_path = os.path.join(label_dir, crop_file)

        try:
            # Load and preprocess image
            image = Image.open(crop_path).convert("RGB")
            image_tensor = transform(np.array(image)).unsqueeze(0).to(device)

            # Extract features using your CNN
            embedding = extractor.get_embeddings(image_tensor)
            embeddings.append(embedding.squeeze(0))
            valid_files += 1

        except Exception as e:
            print(f"  ⚠️  Error processing {crop_file}: {e}")
            continue

    if len(embeddings) == 0:
        print(f"  ❌ No valid embeddings for label: {label}")
        object_stats[label]['status'] = 'no_valid_embeddings'
        continue

    print(f"  ✅ Processed {valid_files}/{len(crop_files)} images successfully")
    object_stats[label]['valid_images'] = valid_files
    object_stats[label]['status'] = 'success'

    # Average all embeddings for the class
    avg_embedding = torch.stack(embeddings).mean(dim=0)
    avg_embedding = avg_embedding / avg_embedding.norm()

    # Store for later analysis
    all_embeddings.append(avg_embedding.cpu().numpy())
    all_labels.append(label)

    # Compute similarity to text embeddings
    avg_embedding = avg_embedding.unsqueeze(1)  # shape: [512, 1]
    scores = torch.matmul(text_embeddings, avg_embedding).squeeze()  # shape: [num_classes]
    similarity_matrix[i, :] = scores

    # Predicted class
    pred_idx = torch.argmax(scores).item()
    pred = text_labels[pred_idx]
    confidence = scores[pred_idx].item()

    print(f"  Predicted: {pred} (confidence: {confidence:.3f})")
    object_stats[label]['prediction'] = pred
    object_stats[label]['confidence'] = confidence
    object_stats[label]['correct'] = pred == label

    if pred == label:
        acc += 1
    total += 1

# ======= RESULTS AND ANALYSIS =======
print("\\n" + "=" * 50)
print("ANALYSIS RESULTS")
print("=" * 50)

if total > 0:
    accuracy = acc / total
    print(f"Overall Accuracy: {accuracy:.3f} ({acc}/{total})")
else:
    print("No valid objects could be processed.")
    exit(1)

# Print detailed statistics
print("\\nDetailed Object Statistics:")
print("-" * 50)
for obj, stats in object_stats.items():
    status = stats.get('status', 'unknown')
    if status == 'success':
        correct = "✅" if stats['correct'] else "❌"
        print(f"{obj:15} | {stats['num_images']:3} images | {stats['valid_images']:3} valid | "
              f"Pred: {stats['prediction']:15} | Conf: {stats['confidence']:.3f} | {correct}")
    else:
        print(f"{obj:15} | Status: {status}")

# ======= VISUALIZATIONS =======
print("\\nGenerating visualizations...")

# 1. Confusion Matrix / Similarity Matrix
plt.figure(figsize=(max(12, len(text_labels)), max(10, len(text_labels))))
similarity_matrix_np = similarity_matrix.cpu().numpy()

sns.heatmap(similarity_matrix_np,
            xticklabels=text_labels,
            yticklabels=text_labels,
            cmap="viridis",
            annot=True,
            fmt=".2f",
            cbar_kws={"label": "Cosine Similarity"})
plt.xlabel("Text Embedding (CLIP)")
plt.ylabel("True Object Class (CNN Features)")
plt.title("CNN-to-CLIP Similarity Matrix")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("cnn_clip_similarity_matrix.png", dpi=300, bbox_inches='tight')
print("✅ Saved: cnn_clip_similarity_matrix.png")

# 2. Accuracy per object
successful_objects = [obj for obj, stats in object_stats.items() if stats.get('status') == 'success']
correct_predictions = [object_stats[obj]['correct'] for obj in successful_objects]
confidences = [object_stats[obj]['confidence'] for obj in successful_objects]

plt.figure(figsize=(max(10, len(successful_objects) * 0.5), 6))
colors = ['green' if correct else 'red' for correct in correct_predictions]
bars = plt.bar(successful_objects, confidences, color=colors, alpha=0.7)
plt.xlabel("Object")
plt.ylabel("Prediction Confidence")
plt.title("Prediction Confidence by Object")
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

# Add accuracy annotation
for i, (obj, conf, correct) in enumerate(zip(successful_objects, confidences, correct_predictions)):
    plt.text(i, conf + 0.01, '✓' if correct else '✗', ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.savefig("object_confidence_analysis.png", dpi=300, bbox_inches='tight')
print("✅ Saved: object_confidence_analysis.png")

# 3. Save detailed results to CSV
results_df = pd.DataFrame([
    {
        'object': obj,
        'num_images': stats.get('num_images', 0),
        'valid_images': stats.get('valid_images', 0),
        'prediction': stats.get('prediction', 'N/A'),
        'confidence': stats.get('confidence', 0.0),
        'correct': stats.get('correct', False),
        'status': stats.get('status', 'unknown')
    }
    for obj, stats in object_stats.items()
])

results_df.to_csv('object_analysis_results.csv', index=False)
print("✅ Saved: object_analysis_results.csv")

# 4. Summary statistics
print("\\nSummary Statistics:")
print("-" * 30)
total_objects = len(all_objects)
processed_objects = len([s for s in object_stats.values() if s.get('status') == 'success'])
total_images = sum(s.get('num_images', 0) for s in object_stats.values())
valid_images = sum(s.get('valid_images', 0) for s in object_stats.values())

print(f"Total objects discovered: {total_objects}")
print(f"Successfully processed: {processed_objects}")
print(f"Total images found: {total_images}")
print(f"Valid images processed: {valid_images}")
print(f"Overall accuracy: {accuracy:.3f}")

# 5. Save summary to JSON
summary = {
    'total_objects': total_objects,
    'processed_objects': processed_objects,
    'total_images': total_images,
    'valid_images': valid_images,
    'accuracy': accuracy,
    'object_details': dict(object_stats)
}

with open('analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("✅ Saved: analysis_summary.json")

print("\\n" + "=" * 50)
print("ANALYSIS COMPLETE!")
print("=" * 50)
print("Generated files:")
print("- cnn_clip_similarity_matrix.png")
print("- object_confidence_analysis.png")
print("- object_analysis_results.csv")
print("- analysis_summary.json")