import os
import json
import cv2
from tqdm import tqdm
from PIL import Image
import numpy as np
import geopandas as gpd
from shapely.geometry import box
import shutil

train_image_dir = r"./train/PS-RGB_tiled"
train_geojson_dir = r"./train/geojson_aircraft_tiled"
test_image_dir = r"./test/PS-RGB_tiled"
test_geojson_dir = r"./test/geojson_aircraft_tiled"

output_dir = './processed'
coco_train_path = os.path.join(output_dir, 'train_coco.json')
coco_test_path = os.path.join(output_dir, 'test_coco.json')

os.makedirs(output_dir, exist_ok=True)

# Step 1: Analyze Dataset
def analyze_dataset():
    print("Analyzing Dataset...")
    print("Training Images:", len(os.listdir(train_image_dir)))
    print("Training GeoJSON Files:", len(os.listdir(train_geojson_dir)))
    print("Testing Images:", len(os.listdir(test_image_dir)))
    print("Testing GeoJSON Files:", len(os.listdir(test_geojson_dir)))

    sample_image_path = os.path.join(train_image_dir, os.listdir(train_image_dir)[0])
    image = cv2.imread(sample_image_path)
    if image is not None:
        print("Sample image shape:", image.shape)
    else:
        print("Error: Could not read sample image")

# Step 2: Check if the file is a valid image
def is_image_file(filename):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
    return filename.lower().endswith(valid_extensions)

# Step 3: Check for corrupted images using PIL
def check_image_validity(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False

# Step 4: Preprocess Images
def preprocess_images(image_dir, output_dir, size=(256, 256)):
    os.makedirs(output_dir, exist_ok=True)
    for img_file in tqdm(os.listdir(image_dir)):
        if not is_image_file(img_file):
            print(f"Skipping non-image file: {img_file}")
            continue

        img_path = os.path.join(image_dir, img_file)
        if not check_image_validity(img_path):
            print(f"Corrupted image found: {img_path}")
            continue

        image = cv2.imread(img_path)
        if image is None:
            print(f"Error: Unable to load image at path: {img_path}")
            continue

        resized_image = cv2.resize(image, size)
        output_path = os.path.join(output_dir, img_file)
        cv2.imwrite(output_path, resized_image)


# Step 5: Convert GeoJSON to COCO format
def geojson_to_coco(image_dir, geojson_dir, output_path):
    """Convert GeoJSON annotations to COCO format."""
    # Initialize COCO structure
    coco_annotations = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 2, "name": "aircraft", "supercategory": "vehicle"}]
    }
    annotation_id = 0
    image_id = 0

    # Loop through all images in the directory
    for img_file in tqdm(os.listdir(image_dir)):
        if not is_image_file(img_file):
            continue

        img_path = os.path.join(image_dir, img_file)
        geojson_file = os.path.join(geojson_dir, img_file.replace('.png', '.geojson'))
        
        if not os.path.exists(geojson_file):
            print(f"GeoJSON not found for {img_file}, skipping...")
            continue

        # Load image to get its dimensions
        image = cv2.imread(img_path)
        if image is None:
            print(f"Error loading image {img_file}, skipping...")
            continue
        height, width, _ = image.shape

        # Add image information
        image_id += 1
        coco_annotations["images"].append({
            "id": image_id,
            "file_name": img_file,
            "width": width,
            "height": height
        })

        # Read the corresponding GeoJSON file
        try:
            with open(geojson_file, 'r') as f:
                geojson_data = json.load(f)
        except Exception as e:
            print(f"Error reading GeoJSON {geojson_file}: {e}")
            continue

        # Iterate through the features in the GeoJSON file
        for feature in geojson_data.get('features', []):
            if feature.get('geometry') is None or feature['geometry']['type'] != 'Polygon':
                continue
            
            # Extract polygon coordinates
            coordinates = feature['geometry']['coordinates'][0]  # Get the first ring of the polygon
            flat_coordinates = [coord for point in coordinates for coord in point]

            # Calculate bounding box from coordinates
            xs = [point[0] for point in coordinates]
            ys = [point[1] for point in coordinates]
            minx, miny, maxx, maxy = min(xs), min(ys), max(xs), max(ys)
            bbox = [minx, miny, maxx - minx, maxy - miny]
            area = bbox[2] * bbox[3]

            # Add annotation linked to the current image
            annotation_id += 1
            coco_annotations["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,  # Link annotation to the correct image
                "category_id": 2,      # The category ID should match the categories section
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
                "segmentation": [flat_coordinates]
            })
    
    # Save to output JSON file
    with open(output_path, 'w') as json_file:
        json.dump(coco_annotations, json_file, indent=4)
    print(f"COCO annotations saved to {output_path}")
    
    
# Step 6: Move auxiliary files
def move_aux_files(src_dir, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    for file in os.listdir(src_dir):
        if file.endswith('.aux.xml'):
            shutil.move(os.path.join(src_dir, file), os.path.join(dest_dir, file))
            print(f"Moved: {file}")

# Step 7: Run the analysis, preprocessing, and annotation conversion
if __name__ == "__main__":
    # Analyze dataset
    # analyze_dataset()

    # Preprocess images
    # preprocess_images(train_image_dir, './processed/train_images')
    # preprocess_images(test_image_dir, './processed/test_images')

    # Convert GeoJSON to COCO format for training and testing sets
    geojson_to_coco(train_image_dir, train_geojson_dir, coco_train_path)
    geojson_to_coco(test_image_dir, test_geojson_dir, coco_test_path)

    # Move .aux.xml files to a separate folder
    move_aux_files(train_geojson_dir, './aux_files/train/')
    move_aux_files(test_geojson_dir, './aux_files/test/')
