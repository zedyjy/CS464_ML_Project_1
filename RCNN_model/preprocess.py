import os
import numpy as np
import cv2
from tqdm import tqdm
import json
import torch
from torchvision.transforms import functional as F
import random

# Data Paths
COCO_ANNOTATIONS_PATH = "path/to/annotations.json"
IMAGES_DIR = "path/to/images"
OUTPUT_DIR = "output/rcnn_preprocessed"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "train_images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "train_annotations"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "val_images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "val_annotations"), exist_ok=True)

# Data Augmentation Functions
def resize_image(image, size=(512, 512)):
    return cv2.resize(image, size)

def random_flip(image, bboxes):
    if random.random() > 0.5:
        image = cv2.flip(image, 1)
        bboxes[:, [0, 2]] = 1 - bboxes[:, [2, 0]]
    if random.random() > 0.5:
        image = cv2.flip(image, 0)
        bboxes[:, [1, 3]] = 1 - bboxes[:, [3, 1]]
    return image, bboxes

def adjust_brightness(image):
    factor = 0.5 + random.random()
    return np.clip(image * factor, 0, 255).astype(np.uint8)

def random_rotate(image, bboxes):
    angle = random.choice([90, 180, 270])
    if angle == 90:
        image = np.rot90(image, 1)
        bboxes = bboxes[:, [1, 0, 3, 2]]
    elif angle == 180:
        image = np.rot90(image, 2)
        bboxes = bboxes[:, [2, 3, 0, 1]]
    elif angle == 270:
        image = np.rot90(image, 3)
        bboxes = bboxes[:, [3, 2, 1, 0]]
    return image, bboxes

# Feature Extraction Stub
def extract_features(image):
    return torch.tensor(image).unsqueeze(0)

# Loss Functions Stub
def calculate_losses(predictions, targets):
    cls_loss = torch.nn.CrossEntropyLoss()(predictions["cls"], targets["cls"])
    reg_loss = torch.nn.SmoothL1Loss()(predictions["bboxes"], targets["bboxes"])
    return cls_loss, reg_loss

# Annotation Conversion to COCO Format
def convert_to_coco_format(images, annotations):
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "aircraft"}
        ]
    }

    annotation_id = 1
    for image_id, (image_path, bboxes) in enumerate(zip(images, annotations)):
        image = cv2.imread(image_path)
        height, width, _ = image.shape

        # Add image info
        coco_data["images"].append({
            "id": image_id,
            "file_name": os.path.basename(image_path),
            "height": height,
            "width": width
        })

        # Add annotations
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                "area": (x_max - x_min) * (y_max - y_min),
                "iscrowd": 0
            })
            annotation_id += 1

    return coco_data

# Data Processing Pipeline
def process_coco_annotations():
    with open(COCO_ANNOTATIONS_PATH, 'r') as f:
        annotations = json.load(f)

    train_data = []
    val_data = []
    for img_id, img_data in tqdm(annotations["images"].items(), desc="Processing images"):
        image_path = os.path.join(IMAGES_DIR, img_data["file_name"])
        if not os.path.exists(image_path):
            continue

        image = cv2.imread(image_path)
        bboxes = np.array(img_data["annotations"])

        # Resize
        image = resize_image(image)

        # Augmentations
        image, bboxes = random_flip(image, bboxes)
        image = adjust_brightness(image)
        image, bboxes = random_rotate(image, bboxes)

        # Split Data
        if random.random() > 0.8:
            split = "val"
            val_data.append((image_path, bboxes))
        else:
            split = "train"
            train_data.append((image_path, bboxes))

        # Save Processed Data
        save_path = os.path.join(OUTPUT_DIR, f"{split}_images", img_data["file_name"])
        cv2.imwrite(save_path, image)

    # Convert to COCO format
    train_coco = convert_to_coco_format(
        [data[0] for data in train_data], [data[1] for data in train_data]
    )
    val_coco = convert_to_coco_format(
        [data[0] for data in val_data], [data[1] for data in val_data]
    )

    # Save COCO annotations
    with open(os.path.join(OUTPUT_DIR, "train_annotations.json"), "w") as f:
        json.dump(train_coco, f)

    with open(os.path.join(OUTPUT_DIR, "val_annotations.json"), "w") as f:
        json.dump(val_coco, f)

    return train_data, val_data

train_data, val_data = process_coco_annotations()

# Feature Extraction Example
for image_path, bboxes in tqdm(train_data, desc="Extracting features"):
    image = cv2.imread(image_path)
    features = extract_features(image)
    # Do something with features

print("Data processing completed!")
