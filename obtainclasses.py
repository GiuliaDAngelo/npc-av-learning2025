import torch
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from PIL import Image
from load_model import EmbeddingExtractor

# ======= CONFIG =======
device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available()
else "cpu")

bbox_dir = "/Users/giuliadangelo/Downloads/npc-av-learning/CRIB/train_data/bbox/"


def get_model_classes():
    """Get the actual classes the model was trained on"""
    try:
        training_info_path = "autoencoder-classifier-trained/training_info.json"
        if os.path.exists(training_info_path):
            with open(training_info_path, 'r') as f:
                training_info = json.load(f)
            model_classes = training_info.get('class_names', [])
            print(f"✅ Found model classes from training info: {model_classes}")
            return model_classes
        else:
            print("⚠️ Training info not found, getting classes from directory structure")
    except Exception as e:
        print(f"⚠️ Error reading training info: {e}")

    # Fallback: get from directory structure
    model_classes = sorted([d for d in os.listdir(bbox_dir)
                            if os.path.isdir(os.path.join(bbox_dir, d)) and not d.startswith('.')])
    print(f"📁 Using classes from directory: {model_classes}")
    return model_classes


def evaluate_model_simple():
    """Simple evaluation focusing on classification accuracy"""

    print(f"🚀 SIMPLE MODEL EVALUATION")
    print(f"Using device: {device}")

    # Load model
    try:
        model = EmbeddingExtractor()
        print("✅ EmbeddingExtractor loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Get classes
    class_names = get_model_classes()
    print(f"📋 Found {len(class_names)} classes: {class_names}")

    if len(class_names) == 0:
        print("❌ No classes found!")
        return

    # Collect embeddings and calculate class centroids
    print(f"\n🔍 STEP 1: Computing class centroids...")
    class_centroids = {}
    class_stats = {}

    for class_name in class_names:
        class_dir = os.path.join(bbox_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"❌ Missing directory for class: {class_name}")
            continue

        # Get image files
        image_files = [f for f in os.listdir(class_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if len(image_files) == 0:
            print(f"⚠️ No images found for class: {class_name}")
            continue

        # Sample up to 30 images per class for speed
        sample_size = min(30, len(image_files))
        sampled_files = np.random.choice(image_files, sample_size, replace=False)

        embeddings = []
        processed = 0
        errors = 0

        print(f"   Processing {class_name}... ", end="")

        for img_file in sampled_files:
            img_path = os.path.join(class_dir, img_file)
            try:
                image = Image.open(img_path).convert("RGB")
                image_array = np.array(image)
                embedding = model.get_embeddings(image_array)
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
            print(f"✅ {processed} images processed (errors: {errors})")
        else:
            print(f"❌ No valid embeddings")

    if len(class_centroids) < 2:
        print("❌ Not enough classes with valid data!")
        return

    # Evaluate classification accuracy
    print(f"\n🎯 STEP 2: Testing classification accuracy...")

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

    # Create the most important visualization: Classification Accuracy
    print(f"\n📊 STEP 3: Creating accuracy visualization...")

    plt.figure(figsize=(8, 8))

    # Main plot: Per-class accuracy
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())

    # Color bars based on performance
    colors = ['green' if acc > 0.8 else 'orange' if acc > 0.6 else 'red' for acc in accuracies]

    bars = plt.bar(range(len(classes)), accuracies, color=colors, alpha=0.7, width=0.2)
    plt.xticks(range(len(classes)), classes, rotation=45, ha='right', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.title(f'Per-Class Classification Accuracy\nOverall: {overall_accuracy:.3f} ({total_correct}/{total_samples})')
    plt.ylim(0, 1)

    # Add accuracy labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{acc:.2f}', ha='center', va='bottom', fontsize=15)

    # Add horizontal line at 80% (good performance threshold)
    plt.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Good Performance (80%)')
    plt.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Fair Performance (60%)')
    plt.legend()

    # # Bottom plot: Sample sizes
    # plt.subplot(2, 1, 2)
    # sample_counts = [class_stats[cls]['processed'] for cls in classes]
    # plt.bar(range(len(classes)), sample_counts, alpha=0.7, color='skyblue')
    # plt.xticks(range(len(classes)), classes, rotation=45, ha='right', fontsize=20)
    # plt.ylabel('Samples Used', fontsize=20)
    # plt.title('Number of Samples per Class')

    plt.tight_layout()
    plt.savefig('model_accuracy_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ======= FINAL ASSESSMENT =======
    print(f"\n{'=' * 60}")
    print("🎯 FINAL ASSESSMENT")
    print(f"{'=' * 60}")

    print(f"📊 Overall Accuracy: {overall_accuracy:.3f} ({total_correct}/{total_samples})")

    # Performance assessment
    if overall_accuracy > 0.8:
        print("   🟢 EXCELLENT: Model has learned very well!")
        verdict = "EXCELLENT"
    elif overall_accuracy > 0.6:
        print("   🟡 GOOD: Model has learned reasonably well")
        verdict = "GOOD"
    elif overall_accuracy > 0.4:
        print("   🟠 FAIR: Model has learned somewhat, but needs improvement")
        verdict = "FAIR"
    else:
        print("   🔴 POOR: Model has not learned well, consider retraining")
        verdict = "POOR"

    # Class-specific insights
    best_classes = [cls for cls, acc in class_accuracies.items() if acc > 0.8]
    worst_classes = [cls for cls, acc in class_accuracies.items() if acc < 0.5]

    if best_classes:
        print(f"   🏆 Best learned classes: {', '.join(best_classes)}")
    if worst_classes:
        print(f"   ⚠️ Poorly learned classes: {', '.join(worst_classes)}")

    # Save simple results
    simple_results = {
        'overall_accuracy': overall_accuracy,
        'verdict': verdict,
        'class_accuracies': class_accuracies,
        'total_samples': total_samples,
        'best_classes': best_classes,
        'worst_classes': worst_classes
    }

    with open('simple_model_evaluation.json', 'w') as f:
        json.dump(simple_results, f, indent=2)

    print("✅ Results saved to 'simple_model_evaluation.json'")
    print("✅ Chart saved to 'model_accuracy_analysis.png'")

    return simple_results


if __name__ == "__main__":
    results = evaluate_model_simple()