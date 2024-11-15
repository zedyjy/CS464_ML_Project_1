import os
import json
import cv2
from tqdm import tqdm
from PIL import Image
import numpy as np
import geopandas as gpd
from shapely.geometry import box
import shutil
from shapely.geometry import shape


train_image_dir = r"C:/Users/FURKAN/train/PS-RGB_tiled"
train_geojson_dir = r"C:/Users/FURKAN/train/geojson_aircraft_tiled"
#test_image_dir = r"C:/Users/FURKAN/test/PS-RGB_tiled"
#test_geojson_dir = r"C:/Users/FURKAN/test/geojson_aircraft_tiled"

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

def geojson_to_coco(image_dir, geojson_dir, output_path):
    """
    Convert GeoJSON annotations to COCO-like hierarchical format, grouping annotations by image.

    Args:
        image_dir (str): Path to the directory containing images.
        geojson_dir (str): Path to the directory containing GeoJSON files.
        output_path (str): Path to save the output COCO JSON file.
    """
    hierarchical_format = {}

    # Define categories (these will be repeated for each image if needed)
    categories = [{"id": 1, "name": "aircraft", "supercategory": "object"}]

    annotation_id = 1
    image_id = 1  # Start with an incremental image ID
    for image_file in tqdm(os.listdir(image_dir)):
        if not is_image_file(image_file):
            continue

        # Image metadata
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)
        if image is None:
            continue
        height, width, _ = image.shape

        # Add image entry with categories included
        hierarchical_format[image_id] = {
            "id": image_id,
            "file_name": image_file,
            "height": height,
            "width": width,
            "categories": categories,  # Add categories here
            "annotations": []
        }

        # Corresponding GeoJSON
        geojson_file = os.path.join(geojson_dir, f"{os.path.splitext(image_file)[0]}.geojson")
        if not os.path.exists(geojson_file):
            continue

        with open(geojson_file, 'r') as f:
            geojson_data = json.load(f)

        # Process features in GeoJSON
        for feature in geojson_data.get("features", []):
            geometry = feature.get("geometry")
            if geometry is None or geometry["type"].lower() != "polygon":
                continue

            polygon = shape(geometry)
            if not polygon.is_valid:
                continue

            segmentation = [list(np.array(polygon.exterior.coords).ravel())]
            x_min, y_min, x_max, y_max = polygon.bounds
            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

            # Add annotation to the image entry
            hierarchical_format[image_id]["annotations"].append({
                "id": annotation_id,
                "category_id": 1,
                "segmentation": segmentation,
                "area": polygon.area,
                "bbox": bbox,
                "iscrowd": 0
            })
            annotation_id += 1

        # Increment the image ID for the next file
        image_id += 1

    # Convert the dictionary format to a list of images
    coco_output = {
        "images": list(hierarchical_format.values())
    }

    # Save to file
    with open(output_path, 'w') as f:
        json.dump(coco_output, f, indent=4)
    print(f"COCO-like hierarchical annotations saved to {output_path}")


    
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
    #geojson_to_coco(test_image_dir, test_geojson_dir, coco_test_path)

    # Move .aux.xml files to a separate folder
    move_aux_files(train_geojson_dir, './aux_files/train/')
    #move_aux_files(test_geojson_dir, './aux_files/test/')
