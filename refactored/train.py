import sys
sys.path.append("../")

import numpy as np
import torch
import cv2
import os
import tqdm
from PIL import Image
import sspspace
import torchvision.transforms as T

import dataset
import saliency
import memory
import embeddings

# Device setup
device = torch.device("mps" if torch.backends.mps.is_available()
                        else "cuda" if torch.cuda.is_available()
else "cpu")
print(f"Using device: {device}")

# Enhanced transforms that match your training data
event_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),  # Match training size
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Match training normalization
])

transform = T.Compose([
    T.Grayscale(),
    T.ToTensor(),
])



# Paths
img_width, img_height = 400, 400
root = '/media/matt/bigdata/DATA/CRIB/train_data/'


# Get all objects from your data
all_objects, all_sequences = dataset.discover_all_objects(root)
print(f"Found {len(all_objects)}")
print(f"Found {len(all_sequences)}")

def learn_object(sequence):
    name = sequence[0].split("/")[-3]
    print(f"Learning object: {name}")

    # [x] generate saccades and get image patches
    # test on one sequence
    ittkoch = saliency.IttiKochNieburSaliency(img_width, img_height)

    cv2.namedWindow("Image", cv2.WINDOW_AUTOSIZE)
    fixations = []
    patch_images = []
    patch_strengths = []
    for img_path in tqdm.tqdm(sequence):
        # load img to numpy array
        img = Image.open(img_path)
        img = np.array(img)
        saliency_map = ittkoch.get_saliency_map(img)
        # get next fixation
        fixation, attention_strength = ittkoch.get_next_fixation(img)
        #print(f"Next fixation: {fixation}")
        patch = embeddings.get_image_patch(img, fixation, size=64)

        # display the image with the patch outlined on it with opencv
        cv2.rectangle(img, (fixation[0]-32, fixation[1]-32), (fixation[0]+32, fixation[1]+32), (255, 0, 0), 2)
        # convert grayscale saliency_map to color and change from range 0-1 to 0-255
        saliency_map = cv2.cvtColor(saliency_map, cv2.COLOR_GRAY2RGB)
        saliency_map = (saliency_map * 255).astype(np.uint8)
        cv2.rectangle(saliency_map, (fixation[0]-32, fixation[1]-32), (fixation[0]+32, fixation[1]+32), (255, 0, 0), 2)
        # concatenate both images and show a single window (800x400)
        combined = np.hstack((img, saliency_map))
        cv2.imshow("Image", combined)
        cv2.waitKey(1)

        fixations.append(fixation)
        patch_images.append(patch)
        patch_strengths.append(attention_strength)

    cv2.destroyAllWindows()


    # [x] get embeddings
    clip_embeddings = embeddings.CLIPEmbeddings(device=device)
    patch_embeddings = clip_embeddings.get_embeddings(patch_images)
    print("Got embeddings of shape", patch_embeddings.shape)


    # [x] learn object with VSA operations
    # Initialize coordinate encoder
    coord_encoder = sspspace.RandomSSPSpace(domain_dim=2, ssp_dim=512)
    # Initialize enhanced memory manager
    memory_manager = memory.WorkingMemoryManager(
        memory_size=20,
        similarity_threshold=0.80,
        min_attention_threshold=80
    )

    for frame_idx, (fixation, patch_image, patch_embedding, attention_strength) in enumerate(zip(fixations, patch_images, patch_embeddings, patch_strengths)):
        # Check if we should store this frame in working memory
        should_store, quality_score, criteria = memory_manager.should_store_memory(
            patch_image, attention_strength, patch_embedding, frame_idx)

        if should_store:
            (x,y) = fixation
            memory_manager.update_memory(patch_embedding, [x, y], quality_score, frame_idx)
            print(f"Frame {frame_idx:3d}: Stored (quality: {quality_score:.3f}, "
                    f"attention: {attention_strength:6.1f}) - {criteria}")

    # Get final consolidated memory
    final_memory = memory_manager.get_consolidated_memory(coord_encoder)
    print("Final memory:", final_memory.shape)
    # save the memory to the memories folder
    np.save(f"memories/{name}_memory.npy", final_memory)

    stats = memory_manager.get_statistics()
    print(stats)


if __name__ == "__main__":
    for seq in all_sequences:
        learn_object(seq)


# [x] learn a working memory for two objects and save it out
# [x] see if we can differentiate between the two based on the training data
# [x] convert the videos to events
# [-] use Giulia's OMS attention model to create fixations instead of the Itti-Koch model
# [ ] create more training and testing data
# [ ] read STAR-FC in more detail and other video-based attention algorithms to compare against
