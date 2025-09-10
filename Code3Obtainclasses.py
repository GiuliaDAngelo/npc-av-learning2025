"""
Model Evaluation Script for Event Frame Autoencoder
Evaluates the trained model using the working memory approach
"""

import torch
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from load_model import EmbeddingExtractor

# Configuration
device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available()
else "cpu")

# Updated paths to match your setup
MODEL_PATH = "/Users/giuliadangelo/workspace/data/DATASETs/CRIB/CRIB400/train_data/resultsbbox30050epochs/final_model/model.pth"
INFO_PATH = "/Users/giuliadangelo/workspace/data/DATASETs/CRIB/CRIB400/train_data/resultsbbox30050epochs/final_model/training_info.json"
BBOX_DIR = "/Users/giuliadangelo/workspace/data/DATASETs/CRIB/CRIB400/train_data/bboxes/"
MEMORY_DIR = "/Users/giuliadangelo/workspace/data/DATASETs/CRIB/CRIB400/train_data/workingmemorybbox30050epochs/"


def get_model_classes():
    """Get the actual classes the model was trained on"""
    try:
        if os.path.exists(INFO_PATH):
            with open(INFO_PATH, 'r') as f:
                training_info = json.load(f)
            model_classes = training_info.get('class_names', [])
            if model_classes:
                print(f"Found model classes from training info: {model_classes}")
                return model_classes
        print("Training info not found or no class names, using directory structure")
    except Exception as e:
        print(f"Error reading training info: {e}")

    # Fallback: get from directory structure
    if os.path.exists(BBOX_DIR):
        model_classes = sorted([d for d in os.listdir(BBOX_DIR)
                                if os.path.isdir(os.path.join(BBOX_DIR, d)) and not d.startswith('.')])
        print(f"Using classes from bbox directory: {model_classes}")
        return model_classes

    print("No class information available")
    return []


def evaluate_working_memory_approach():
    """Evaluate using the working memory files created by your processing pipeline"""

    print("Working Memory Evaluation")
    print(f"Using device: {device}")

    # Check if memory directory exists
    if not os.path.exists(MEMORY_DIR):
        print(f"Memory directory not found: {MEMORY_DIR}")
        return None

    # Get memory files
    memory_files = [f for f in os.listdir(MEMORY_DIR) if f.endswith('_memory.npy')]
    feature_files = [f for f in os.listdir(MEMORY_DIR) if f.endswith('_image_features.npy')]

    print(f"Found {len(memory_files)} memory files and {len(feature_files)} feature files")

    if len(memory_files) == 0:
        print("No memory files found. Run the processing pipeline first.")
        return None

    # Load working memories and features
    object_memories = {}
    object_features = {}

    for memory_file in memory_files:
        obj_name = memory_file.replace('_memory.npy', '')
        memory_path = os.path.join(MEMORY_DIR, memory_file)
        feature_path = os.path.join(MEMORY_DIR, f'{obj_name}_image_features.npy')

        try:
            memory = np.load(memory_path)
            object_memories[obj_name] = memory

            if os.path.exists(feature_path):
                features = np.load(feature_path)
                object_features[obj_name] = features

            print(f"  Loaded {obj_name}: memory shape {memory.shape}")

        except Exception as e:
            print(f"  Error loading {obj_name}: {e}")

    if len(object_memories) < 2:
        print("Need at least 2 objects for evaluation")
        return None

    print(f"\nLoaded working memories for {len(object_memories)} objects")
    return object_memories, object_features


def evaluate_bbox_classification():
    """Evaluate classification accuracy using bbox data with proper event frame preprocessing"""

    print("Bbox Classification Evaluation")

    # Load model
    try:
        model = EmbeddingExtractor(model_path=MODEL_PATH, info_path=INFO_PATH)
        print("EmbeddingExtractor loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    # Get classes
    class_names = get_model_classes()
    if len(class_names) == 0:
        print("No classes found")
        return None

    print(f"Found {len(class_names)} classes: {class_names}")

    # Collect embeddings and calculate class centroids
    print("\nComputing class centroids...")
    class_centroids = {}
    class_stats = {}

    for class_name in class_names:
        class_dir = os.path.join(BBOX_DIR, class_name)
        if not os.path.exists(class_dir):
            print(f"Missing directory for class: {class_name}")
            continue

        # Get image files
        image_files = [f for f in os.listdir(class_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if len(image_files) == 0:
            print(f"No images found for class: {class_name}")
            continue

        # Sample images for speed
        sample_size = min(20, len(image_files))
        sampled_files = np.random.choice(image_files, sample_size, replace=False)

        embeddings = []
        processed = 0
        errors = 0

        print(f"   Processing {class_name}... ", end="")

        for img_file in sampled_files:
            img_path = os.path.join(class_dir, img_file)
            try:
                # Load and preprocess as grayscale (event frame format)
                image = Image.open(img_path).convert("L")  # Convert to grayscale
                image_array = np.array(image)

                # Resize if needed
                if image_array.shape != (224, 224):
                    image_array = cv2.resize(image_array, (224, 224), interpolation=cv2.INTER_LINEAR)

                # Get embeddings
                embedding = model.get_embeddings(image_array)

                # Normalize embedding
                embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
                embeddings.append(embedding)
                processed += 1

            except Exception as e:
                errors += 1

        if len(embeddings) > 0:
            embeddings = np.array(embeddings)
            centroid = np.mean(embeddings, axis=0)
            class_centroids[class_name] = centroid
            class_stats[class_name] = {
                'embeddings': embeddings,
                'processed': processed,
                'errors': errors,
                'total_files': len(image_files)
            }
            print(f"processed {processed} images (errors: {errors})")
        else:
            print("no valid embeddings")

    if len(class_centroids) < 2:
        print("Not enough classes with valid data")
        return None

    # Evaluate classification accuracy
    print("\nTesting classification accuracy...")

    class_accuracies = {}
    total_correct = 0
    total_samples = 0

    for true_class, stats in class_stats.items():
        if true_class not in class_centroids:
            continue

        embeddings = stats['embeddings']
        correct = 0

        for embedding in embeddings:
            # Find nearest centroid
            min_distance = float('inf')
            predicted_class = None

            for centroid_class, centroid in class_centroids.items():
                distance = np.linalg.norm(embedding - centroid)
                if distance < min_distance:
                    min_distance = distance
                    predicted_class = centroid_class

            if predicted_class == true_class:
                correct += 1
                total_correct += 1
            total_samples += 1

        accuracy = correct / len(embeddings) if len(embeddings) > 0 else 0
        class_accuracies[true_class] = accuracy
        print(f"   {true_class:15s}: {accuracy:.3f} ({correct}/{len(embeddings)})")

    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0

    # Visualization
    create_accuracy_visualization(class_accuracies, overall_accuracy, total_correct, total_samples)

    # Assessment
    print_final_assessment(overall_accuracy, class_accuracies, total_correct, total_samples)

    return {
        'overall_accuracy': overall_accuracy,
        'class_accuracies': class_accuracies,
        'total_samples': total_samples,
        'class_stats': class_stats
    }


def create_accuracy_visualization(class_accuracies, overall_accuracy, total_correct, total_samples):
    """Create accuracy visualization"""

    plt.figure(figsize=(12, 6))

    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())

    # Color bars based on performance
    colors = ['green' if acc > 0.8 else 'orange' if acc > 0.6 else 'red' for acc in accuracies]

    bars = plt.bar(range(len(classes)), accuracies, color=colors, alpha=0.7)
    plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
    plt.ylabel('Accuracy')
    plt.title(f'Per-Class Classification Accuracy\nOverall: {overall_accuracy:.3f} ({total_correct}/{total_samples})')
    plt.ylim(0, 1)

    # Add accuracy labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{acc:.2f}', ha='center', va='bottom')

    # Add performance threshold lines
    plt.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Good (80%)')
    plt.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Fair (60%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('model_accuracy_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()


def print_final_assessment(overall_accuracy, class_accuracies, total_correct, total_samples):
    """Print final assessment"""

    print(f"\n{'=' * 60}")
    print("FINAL ASSESSMENT")
    print(f"{'=' * 60}")

    print(f"Overall Accuracy: {overall_accuracy:.3f} ({total_correct}/{total_samples})")

    # Performance assessment
    if overall_accuracy > 0.8:
        print("   EXCELLENT: Model has learned very well")
        verdict = "EXCELLENT"
    elif overall_accuracy > 0.6:
        print("   GOOD: Model has learned reasonably well")
        verdict = "GOOD"
    elif overall_accuracy > 0.4:
        print("   FAIR: Model has learned somewhat, but needs improvement")
        verdict = "FAIR"
    else:
        print("   POOR: Model has not learned well, consider retraining")
        verdict = "POOR"

    # Class-specific insights
    best_classes = [cls for cls, acc in class_accuracies.items() if acc > 0.8]
    worst_classes = [cls for cls, acc in class_accuracies.items() if acc < 0.5]

    if best_classes:
        print(f"   Best learned classes: {', '.join(best_classes)}")
    if worst_classes:
        print(f"   Poorly learned classes: {', '.join(worst_classes)}")

    # Save results
    results = {
        'overall_accuracy': overall_accuracy,
        'verdict': verdict,
        'class_accuracies': class_accuracies,
        'total_samples': total_samples,
        'best_classes': best_classes,
        'worst_classes': worst_classes
    }

    with open('model_evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("Results saved to 'model_evaluation_results.json'")
    print("Chart saved to 'model_accuracy_analysis.png'")


def main():
    """Main evaluation function"""

    print("Model Evaluation Pipeline")
    print("=" * 50)

    # Option 1: Evaluate working memory (if available)
    memory_result = evaluate_working_memory_approach()
    if memory_result:
        print("Working memory evaluation completed")

    # Option 2: Evaluate bbox classification
    bbox_result = evaluate_bbox_classification()
    if bbox_result:
        print("Bbox classification evaluation completed")

    if not memory_result and not bbox_result:
        print("No evaluation could be performed. Check your data paths.")


if __name__ == "__main__":
    main()