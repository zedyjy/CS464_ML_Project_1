import os
import json
import xml.etree.ElementTree as ET
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# --------------------------
#  Utility Functions
# --------------------------
def parse_geo_transform(aux_path):
    """
    Parse the GeoTransform metadata from the .aux.xml file.
    """
    tree = ET.parse(aux_path)
    root = tree.getroot()
    geo_transform = root.find("GeoTransform").text.strip().split(",")
    geo_transform = [float(value) for value in geo_transform]
    return geo_transform

def geo_to_pixel(lon, lat, geo_transform):
    """
    Convert geographic coordinates (longitude, latitude) to image pixel coordinates.
    """
    x_origin, pixel_width, _, y_origin, _, pixel_height = geo_transform
    x_pixel = int((lon - x_origin) / pixel_width)
    y_pixel = int((y_origin - lat) / abs(pixel_height))
    return x_pixel, y_pixel

def geojson_to_pixel_bboxes(geojson_path, geo_transform):
    """
    Convert GeoJSON bounding boxes to image pixel bounding boxes.
    """
    with open(geojson_path, 'r') as f:
        data = json.load(f)

    pixel_bboxes = []
    for feature in data['features']:
        coords = feature['geometry']['coordinates'][0]
        pixel_coords = [geo_to_pixel(lon, lat, geo_transform) for lon, lat in coords]
        x_coords = [p[0] for p in pixel_coords]
        y_coords = [p[1] for p in pixel_coords]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        pixel_bboxes.append((x_min, y_min, x_max, y_max))
    return pixel_bboxes

def clamp_bbox(bbox, image_width, image_height):
    """
    Clamp bounding box to ensure it stays within image bounds.
    """
    x_min, y_min, x_max, y_max = bbox
    x_min = max(0, min(image_width, x_min))
    x_max = max(0, min(image_width, x_max))
    y_min = max(0, min(image_height, y_min))
    y_max = max(0, min(image_height, y_max))
    return x_min, y_min, x_max, y_max

# --------------------------
#  Dataset Preprocessing
# --------------------------
def preprocess_data(image_folder, feature_folder, aux_folder, output_image_folder, output_feature_folder, transform):
    """
    Preprocess data: converts GeoJSON coordinates to pixel coordinates,
    normalizes bounding boxes, and saves processed images and features.
    """
    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_feature_folder, exist_ok=True)

    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])
    label_files = sorted([f for f in os.listdir(feature_folder) if f.endswith('.geojson')])

    for img_file, label_file in tqdm(zip(image_files, label_files), total=len(image_files)):
        img_path = os.path.join(image_folder, img_file)
        label_path = os.path.join(feature_folder, label_file)
        aux_path = os.path.join(aux_folder, img_file + '.aux.xml')

        # Parse GeoTransform from .aux.xml
        geo_transform = parse_geo_transform(aux_path)

        # Load image
        image = Image.open(img_path).convert('RGB')
        width, height = image.size

        # Convert GeoJSON to pixel bounding boxes
        pixel_bboxes = geojson_to_pixel_bboxes(label_path, geo_transform)

        # Clamp bounding boxes
        clamped_bboxes = [clamp_bbox(bbox, width, height) for bbox in pixel_bboxes]

        # Normalize bounding boxes
        normalized_bboxes = [
            [
                (x_min + x_max) / (2 * width),  # cx
                (y_min + y_max) / (2 * height),  # cy
                (x_max - x_min) / width,  # w
                (y_max - y_min) / height,  # h
            ]
            for x_min, y_min, x_max, y_max in clamped_bboxes
        ]

        # Save processed image
        processed_image = transform(image)
        torch.save(processed_image, os.path.join(output_image_folder, img_file.replace('.png', '.pt')))

        # Save processed features
        torch.save(torch.tensor(normalized_bboxes), os.path.join(output_feature_folder, label_file.replace('.geojson', '.pt')))

# --------------------------
#  Main Function
# --------------------------
if __name__ == "__main__":
    # Input and output paths
    train_image_folder = "./data/raw/train/PS-RGB_tiled"
    train_aux_folder = "./data/raw/train/PS-RGB_tiled"
    train_feature_folder = "./data/raw/train/geojson_aircraft_tiled"
    test_image_folder = "./data/raw/test/PS-RGB_tiled"
    test_aux_folder = "./data/raw/test/PS-RGB_tiled"
    test_feature_folder = "./data/raw/test/geojson_aircraft_tiled"

    train_output_image_folder = "./data/processed/train/images"
    train_output_feature_folder = "./data/processed/train/features"
    test_output_image_folder = "./data/processed/test/images"
    test_output_feature_folder = "./data/processed/test/features"

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Preprocess training and test data
    print("Preprocessing training data...")
    preprocess_data(
        train_image_folder,
        train_feature_folder,
        train_aux_folder,
        train_output_image_folder,
        train_output_feature_folder,
        transform
    )

    print("Preprocessing test data...")
    preprocess_data(
        test_image_folder,
        test_feature_folder,
        test_aux_folder,
        test_output_image_folder,
        test_output_feature_folder,
        transform
    )
    print("Preprocessing complete.")
