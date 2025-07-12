
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch


# run with mini bbox and big box with gamma 0.99
# save accuracy
# create new dataset where we have same object different movements 3 for each object


# Load model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

text_labels = [
        [
            "bag",
            "baseball",
            "binoculars",
            "cactus",
            "crystal",
            "dozer",
            "drill",
            "ducky",
            "dump_truck",
            "firetruck",
            "car",
            "football",
            "hammer",
            "helicopter",
            "horse",
            "life_belt",
            "motorbike",
            "noise_toy",
            "phone",
            "roman_helmet",
            "stacked_toy",
            "star",
            "stroller",
            "wheel",
            "xylophone"
        ],
        [
            "banana",
            "baymax",
            "bucket",
            "cockroach",
            "coin",
            "crocodile",
            "cupcake",
            "elephant",
            "spinner",
            "frog",
            "helmet",
            "ice_cream",
            "lollipop",
            "mailbox",
            "orangering",
            "soccer_ball",
            "spinning_top",
            "sumo",
            "teacup",
            "teddy",
            "tractor",
            "traffic_cone",
            "treasure_chest",
            "trophy",
            "ufo"
        ],
        [
            "android",
            "badminton_racket",
            "barn",
            "bass",
            "boat",
            "burger",
            "camel",
            "chair",
            "cucumber",
            "cup",
            "dice",
            "domino",
            "dragon",
            "fire_hydrant",
            "fish",
            "food_shop",
            "pear",
            "powerpuff_girl",
            "ring_sphere",
            "rubik",
            "sheep",
            "smurf",
            "sword",
            "van",
            "volleyball"
        ],
        [
            "alien",
            "beachball",
            "blimp",
            "book",
            "cake",
            "cow",
            "monster",
            "nerfgun",
            "panda",
            "picnic_table",
            "plane",
            "plate",
            "pudding",
            "radio",
            "rooster",
            "shovel",
            "skateboard",
            "soda_can",
            "spaceship",
            "teapot",
            "tomato",
            "tricycle",
            "truck",
            "turtle",
            "vase"
        ],
        [
            "bow",
            "baby",
            "balls",
            "basketball",
            "bee",
            "bike_helmet",
            "birdie",
            "bulb",
            "bunny",
            "comb",
            "cookie",
            "dog",
            "dolphin",
            "doraemon",
            "film_clapper",
            "fork",
            "fox",
            "grapes",
            "heart",
            "hotdog",
            "house",
            "knight",
            "lego_block",
            "lego_man",
            "lobster"
        ],
        [
            "penguin",
            "piano",
            "pig",
            "pikachu",
            "pirate_ship",
            "pizza",
            "platypus",
            "ram",
            "shaggy",
            "soldier",
            "solenodon",
            "spiderman",
            "spongebob",
            "spoon",
            "squirrel",
            "submarine",
            "tennis_ball",
            "toothbrush",
            "train",
            "tv",
            "umbrella",
            "vulture",
            "wand",
            "woody",
            "yoyo"
        ],
        [
            "anger",
            "apple",
            "baby_dino",
            "baby_rattle",
            "bardak",
            "hippo",
            "homer",
            "horn",
            "inspector",
            "key",
            "laptop",
            "mickey",
            "minion",
            "monkey",
            "octopus",
            "olaf",
            "orange",
            "pan",
            "pen",
            "police_car",
            "pumpkin",
            "shark",
            "snail",
            "tree",
            "wolfy"
        ],
        [
            "baby_bottle",
            "cat",
            "cheburashka",
            "chess_king",
            "donkey",
            "flower",
            "fork_lift",
            "giraffe",
            "lamp",
            "lego_bicycle",
            "luffy",
            "megaman",
            "monimop",
            "mushroom",
            "pokeball",
            "reindeer",
            "robot",
            "santa",
            "shopping_cart",
            "sneaker",
            "sonic",
            "tank",
            "unicorn",
            "whistle",
            "worm"
        ]
    ]
import numpy as np
text_labels = np.array(text_labels).flatten()
text_labels_og = text_labels.copy()
text_labels = ["a photo of a " + t for t in text_labels]
text_tokens = clip_processor(text=text_labels, return_tensors="pt", padding=True) #check this!!
with torch.no_grad():
    text_embeddings = clip_model.get_text_features(**text_tokens)
    text_embeddings = text_embeddings/text_embeddings.norm(p=2, dim=-1,keepdim=True)


# for over images use match name to text_labels_og
image_embeddings = np.load("/Users/giuliadangelo/Downloads/npc-av-learning/CRIB/workingmemory/alien_memory.npy")  # Load your object memory vectors here
image_embeddings = torch.tensor(image_embeddings, dtype=torch.float32)
image_embeddings = image_embeddings/image_embeddings.norm()

scores = text_embeddings @ image_embeddings.T  # Compute cosine similarity
print(text_labels[np.argmax(scores)])