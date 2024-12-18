import os
import json
import yaml
import cv2
from tqdm import tqdm
from PIL import Image
import numpy as np
import geopandas as gpd
from shapely.geometry import box
import shutil
from shapely.geometry import shape

train_image_dir = r"./raw/train/PS-RGB_tiled"
train_geojson_dir = r"./raw/train/geojson_aircraft_tiled"
test_image_dir = r"./raw/test/PS-RGB_tiled"
test_geojson_dir = r"./raw/test/geojson_aircraft_tiled"

output_dir = './processed'
coco_train_path = os.path.join(output_dir, 'train_coco.yaml')
coco_test_path = os.path.join(output_dir, 'test_coco.yaml')
yolo_train_labels = os.path.join(output_dir, 'train/labels')
yolo_test_labels = os.path.join(output_dir, 'test/labels')
yolo_train_images = os.path.join(output_dir, 'train/images')
yolo_test_images = os.path.join(output_dir, 'test/images')

os.makedirs(output_dir, exist_ok=True)
os.makedirs(yolo_train_labels, exist_ok=True)
os.makedirs(yolo_train_images, exist_ok=True)
os.makedirs(yolo_test_labels, exist_ok=True)
os.makedirs(yolo_test_images, exist_ok=True)

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
            # print(f"Skipping non-image file: {img_file}")
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

def convert_geojson_to_yolo(geojson_dir, image_dir, output_label_dir):
    """Convert GeoJSON annotations to YOLO format."""
    os.makedirs(output_label_dir, exist_ok=True)
    
    for geojson_file in tqdm(os.listdir(geojson_dir)):
        if not geojson_file.endswith('.geojson'):
            continue

        geojson_path = os.path.join(geojson_dir, geojson_file)
        image_name = geojson_file.replace('.geojson', '.png')
        image_path = os.path.join(image_dir, image_name)
        
        if not os.path.exists(image_path):
            print(f"Image not found for {image_name}, skipping...")
            continue
        
        # Load image to get dimensions
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image {image_name}, skipping...")
            continue
        img_height, img_width, _ = image.shape

        # Read GeoJSON file
        try:
            with open(geojson_path, 'r') as f:
                geojson_data = json.load(f)
        except Exception as e:
            print(f"Error reading GeoJSON {geojson_file}: {e}")
            continue

        # Create corresponding YOLO label file
        label_path = os.path.join(output_label_dir, f"{os.path.splitext(image_name)[0]}.txt")
        
        with open(label_path, 'w') as label_file:
            for feature in geojson_data.get('features', []):
                geometry = feature.get('geometry')
                if geometry is None or geometry['type'] != 'Polygon':
                    continue

                # Extract bounding box
                xs = [coord[0] for coord in geometry['coordinates'][0]]
                ys = [coord[1] for coord in geometry['coordinates'][0]]
                x_min, y_min, x_max, y_max = min(xs), min(ys), max(xs), max(ys)

                # Convert to YOLO format
                # Normalize the values to [0, 1]
                center_x = ((x_min + x_max) / 2) / img_width
                center_y = ((y_min + y_max) / 2) / img_height
                width = (x_max - x_min) / img_width
                height = (y_max - y_min) / img_height

                # Ensure that no values are negative or greater than 1
                center_x = max(0, min(1, center_x))
                center_y = max(0, min(1, center_y))
                width = max(0, min(1, width))
                height = max(0, min(1, height))

                # Write annotation to file
                label_file.write(f"0 {center_x} {center_y} {width} {height}\n")

        #print(f"YOLO labels saved to {label_path}")


# Step 6: Move auxiliary files
def move_aux_files(src_dir, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    for file in os.listdir(src_dir):
        if file.endswith('.aux.xml'):
            shutil.move(os.path.join(src_dir, file), os.path.join(dest_dir, file))
            print(f"Moved: {file}")

# Step 7: Run the analysis, preprocessing, and annotation conversion
if __name__ == "__main__":
    analyze_dataset()
    
    # # Preprocess images
    print("Preprocessing train images...")
    preprocess_images(train_image_dir, yolo_train_images)
    print("Preprocessing test images...")
    preprocess_images(test_image_dir, yolo_test_images)

    # Convert GeoJSON to YOLO format
    print("Preprocessing train GeoJSON files...")
    convert_geojson_to_yolo(train_geojson_dir, train_image_dir, yolo_train_labels)
    print("Preprocessing test GeoJSON files...")
    convert_geojson_to_yolo(test_geojson_dir, test_image_dir, yolo_test_labels)

    # Move .aux.xml files to a separate folder
    move_aux_files(train_geojson_dir, './processed/aux_files/train/')
    move_aux_files(test_geojson_dir, './processed/aux_files/test/')
