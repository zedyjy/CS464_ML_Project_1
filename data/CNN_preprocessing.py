import os
import json
import xml.etree.ElementTree as ET
import torch
from random import randint
from PIL import Image, ImageDraw
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image

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
    
    # print(f"GeoTransform: {geo_transform}")
    return geo_transform

def read_geojson(geojson_path):
    """
    Read and parse GeoJSON file.
    """
    with open(geojson_path, 'r') as f:
        data = json.load(f)
        
    if not data['features']:
        coords = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]  # Dummy coordinates
        features = [0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0]
    else:
        first_feature = data['features'][0]
        coords = first_feature['geometry']['coordinates']
        
        # Ensure `coords` is a list of tuples/lists
        if not isinstance(coords, list) or not all(isinstance(pt, (list, tuple)) and len(pt) == 2 for pt in coords[0]):
            raise ValueError(f"Invalid GeoJSON coordinates: {coords}")
        
        coords = coords[0]  # Extract the first polygon
        
        xs = [pt[0] for pt in coords]
        ys = [pt[1] for pt in coords]
        
        if not xs or not ys:
            coords = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]  # Dummy coordinates
            features = [0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0]
        else:
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
        
        # Extract additional properties
        props = first_feature.get('properties', {})
        length = props.get('length', 0.0)
        wingspan = props.get('wingspan', 0.0)
        area = props.get('area', 0.0)

        # Handle categorical features
        wing_type_str = props.get('wing_type', 'other')
        wing_position_str = props.get('wing_position', 'other')

        wing_type_code = 0 if wing_type_str == 'straight' else (1 if wing_type_str == 'swept' else 2)
        wing_position_code = 0 if wing_position_str == 'high mounted' else (1 if 'low' in wing_position_str or 'mid' in wing_position_str else 2)

        canard = 1 if props.get('canards', 'no') == 'yes' else 0
        num_engines = props.get('num_engines', 0)
        num_tailfins = props.get('num_tail_fins', 0)
        faa_class = props.get('faa_wingspan_class', 0)

        features = [length, wingspan, area, wing_type_code, wing_position_code, canard, num_engines, num_tailfins, faa_class]
    
    # print(f"GeoJSON Coords: {coords}")
    # print(f"GeoJSON Features: {features}")
    return coords, features

                

def geo_to_pixel(lon, lat, geo_transform, image_width, image_height):
    """
    Convert geographic coordinates (longitude, latitude) to image pixel coordinates.
    """
    x_origin, pixel_width, _, y_origin, _, pixel_height = geo_transform
    x_pixel = int((lon - x_origin) / pixel_width)
    y_pixel = int((y_origin - lat) / abs(pixel_height))

    # Clamp to image dimensions
    x_pixel = max(0, min(x_pixel, image_width - 1))
    y_pixel = max(0, min(y_pixel, image_height - 1))
    
    # print(f"Pixel Coordinates: x={x_pixel}, y={y_pixel}")
    return x_pixel, y_pixel


def geojson_to_pixel_bboxes(coords, geo_transform, image_width, image_height):
    """
    Convert GeoJSON bounding boxes to image pixel bounding boxes.
    """
    pixel_coords = [geo_to_pixel(lon, lat, geo_transform, image_width, image_height) for lon, lat in coords]
    x_coords = [p[0] for p in pixel_coords]
    y_coords = [p[1] for p in pixel_coords]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # print(f"Pixel Bounding Box: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")
    return x_min, y_min, x_max, y_max


# --------------------------
#  Dataset Preprocessing
# --------------------------
def preprocess_data(image_folder, feature_folder, aux_folder, output_image_file, output_feature_file, transform):
    """
    Preprocess data: extracts the first bounding box and additional features for each image,
    and saves processed images and features in single files.
    """
    os.makedirs(os.path.dirname(output_image_file), exist_ok=True)
    os.makedirs(os.path.dirname(output_feature_file), exist_ok=True)

    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])
    label_files = sorted([f for f in os.listdir(feature_folder) if f.endswith('.geojson')])
    aux_files = sorted([f for f in os.listdir(aux_folder) if f.endswith('.aux.xml')])

    processed_images = []
    processed_features = []

    for img_file, label_file, aux_file in tqdm(zip(image_files, label_files, aux_files), total=len(image_files)):
        img_path = os.path.join(image_folder, img_file)
        label_path = os.path.join(feature_folder, label_file)
        aux_path = os.path.join(aux_folder, aux_file)

        # Parse GeoTransform from .aux.xml
        geo_transform = parse_geo_transform(aux_path)
        
        # Parse GeoJSON and extract bounding box
        coords, features = read_geojson(label_path)

        # Load image
        image = Image.open(img_path).convert('RGB')
        width, height = image.size

        # Convert coords to pixel coords
        pixel_bbox = geojson_to_pixel_bboxes(coords, geo_transform, width, height)

        # Convert corner format to center format and normalize
        x_min, y_min, x_max, y_max = pixel_bbox
        x_min, x_max = x_min / width, x_max / width
        y_min, y_max = y_min / height, y_max / height
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        w = (x_max - x_min)
        h = (y_max - y_min)
        centered_bbox = [cx, cy, w, h]
        # print(f"Center Format: cx={cx}, cy={cy}, w={w}, h={h}")
        
        # Apply transformation to the image
        processed_image = transform(image)

        # Add processed image and combined features to the lists
        processed_images.append(processed_image)
        processed_features.append(torch.tensor(centered_bbox + features, dtype=torch.float32))

    # Stack processed images into a tensor
    torch.save(torch.stack(processed_images), output_image_file)

    # Stack processed features into a tensor
    torch.save(torch.stack(processed_features), output_feature_file)

# --------------------------
#  Visualization Function
# --------------------------
def visualize_first_data(image_file, feature_file):
    """
    Visualize the first image and its corresponding bounding box and features.
    """
    # Load the processed data
    images = torch.load(image_file, weights_only=True)
    features = torch.load(feature_file, weights_only=True)

    # Get the first image and features
    n = randint(0, len(images) - 1)
    first_image = images[n]  # Tensor: [3, 256, 256]
    first_features = features[n]  # Tensor: [cx, cy, w, h, ...other features]

    # Convert tensor image back to PIL image for visualization
    mean = torch.tensor([0.5, 0.5, 0.5])
    std = torch.tensor([0.5, 0.5, 0.5])
    denormalized_image = first_image * std[:, None, None] + mean[:, None, None]
    pil_image = to_pil_image(denormalized_image)

    # Extract bounding box (cx, cy, w, h) and convert to corner format
    cx, cy, w, h = first_features[:4]
    width, height = pil_image.size
    x_min = (cx - w / 2) * width
    x_max = (cx + w / 2) * width
    y_min = (cy - h / 2) * height
    y_max = (cy + h / 2) * height

    # Validate bounding box coordinates
    if x_min < 0 or y_min < 0 or x_max > width or y_max > height or w <= 0 or h <= 0:
        print(f"Invalid bounding box: [{x_min}, {y_min}, {x_max}, {y_max}]")
        return

    # Draw bounding box on the image
    draw = ImageDraw.Draw(pil_image)
    draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)

    # Display the image
    plt.figure(figsize=(6, 6))
    plt.imshow(pil_image)
    plt.title(f"Bounding Box: [{x_min:.2f}, {y_min:.2f}, {x_max:.2f}, {y_max:.2f}]")
    plt.axis("off")
    plt.show()

    # Print additional features
    additional_features = first_features[4:]
    print(f"Additional Features: {additional_features.tolist()}")

# --------------------------
#  Main Function
# --------------------------
if __name__ == "__main__":
    # Input and output paths
    train_image_folder = r"./data/raw/train/PS-RGB_tiled"
    train_aux_folder = r"./data/raw/train/PS-RGB_tiled"
    train_feature_folder = r"./data/raw/train/geojson_aircraft_tiled"
    test_image_folder = r"./data/raw/test/PS-RGB_tiled"
    test_aux_folder = r"./data/raw/test/PS-RGB_tiled"
    test_feature_folder = r"./data/raw/test/geojson_aircraft_tiled"
    train_output_image_file = "./data/processed/train/images.pt"
    train_output_feature_file = "./data/processed/train/features.pt"
    test_output_image_file = "./data/processed/test/images.pt"
    test_output_feature_file = "./data/processed/test/features.pt"

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Process training and test data
    print("Processing training data...")
    preprocess_data(
        train_image_folder,
        train_feature_folder,
        train_aux_folder,
        train_output_image_file,
        train_output_feature_file,
        transform
    )
    print("Processing test data...")
    preprocess_data(
        test_image_folder,
        test_feature_folder,
        test_aux_folder,
        test_output_image_file,
        test_output_feature_file,
        transform
    )
    
    visualize_first_data(train_output_image_file, train_output_feature_file)
    print("Processing complete.")