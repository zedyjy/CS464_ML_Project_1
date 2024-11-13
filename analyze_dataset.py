import os
import cv2
from tqdm import tqdm
from PIL import Image
import numpy as np

train_image_dir = r"C:\Users\zeyne\CS464_ML_Project_1\CS464_ML_Project_1\train\PS-RGB_tiled"
train_geojson_dir = r"C:\Users\zeyne\CS464_ML_Project_1\CS464_ML_Project_1\train\geojson_aircraft_tiled"
test_image_dir = r"C:\Users\zeyne\CS464_ML_Project_1\CS464_ML_Project_1\test\PS-RGB_tiled"
test_geojson_dir = r"C:\Users\zeyne\CS464_ML_Project_1\CS464_ML_Project_1\test\geojson_aircraft_tiled"


# Step 1: Analyze Dataset
def analyze_dataset():
    print("Analyzing Dataset...")
    
    # Check the number of files in each directory
    print("Training Images:", len(os.listdir(train_image_dir)))
    print("Training GeoJSON Files:", len(os.listdir(train_geojson_dir)))
    print("Testing Images:", len(os.listdir(test_image_dir)))
    print("Testing GeoJSON Files:", len(os.listdir(test_geojson_dir)))
    
    # Check the shape of a sample image
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
            img.verify()  # Verify it's a valid image
        return True
    except Exception:
        return False

# Step 4: Preprocess images
def preprocess_images(image_dir, output_dir, size=(256, 256)):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Preprocessing images from {image_dir}...")
    for img_file in tqdm(os.listdir(image_dir)):
        if not is_image_file(img_file):
            print(f"Skipping non-image file: {img_file}")
            continue

        img_path = os.path.join(image_dir, img_file)
        
        # Check if the image is corrupted
        if not check_image_validity(img_path):
            print(f"Corrupted image found: {img_path}")
            continue
        
        # Load image using OpenCV
        image = cv2.imread(img_path)
        
        # Check if image was successfully loaded
        if image is None:
            print(f"Error: Unable to load image at path: {img_path}")
            continue
        
        # Resize the image
        resized_image = cv2.resize(image, size)
        
        # Save the resized image
        output_path = os.path.join(output_dir, img_file)
        cv2.imwrite(output_path, resized_image)

# Step 5: Run the analysis and preprocessing
if __name__ == "__main__":
    # Analyze dataset
    analyze_dataset()
    
    # Preprocess training images
    preprocess_images(train_image_dir, './processed/train_images')
    
    # Preprocess testing images
    preprocess_images(test_image_dir, './processed/test_images')
    
    
import os
import shutil

def move_aux_files(src_dir, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    for file in os.listdir(src_dir):
        if file.endswith('.aux.xml'):
            shutil.move(os.path.join(src_dir, file), os.path.join(dest_dir, file))
            print(f"Moved: {file}")

# Move .aux.xml files to a separate folder
move_aux_files(train_image_dir, './aux_files/train/')
move_aux_files(test_image_dir, './aux_files/test/')
