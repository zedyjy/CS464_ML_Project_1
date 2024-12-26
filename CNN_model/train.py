import os
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from torchvision import transforms
from torchvision import models, ops
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
from datetime import datetime
from tqdm import tqdm

# ---------------------------------------------------------------------
#  Utility Functions
# ---------------------------------------------------------------------
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

def corner_to_center(bboxes):
    """
    Convert bounding boxes from corner format (x_min, y_min, x_max, y_max)
    to center-size format (cx, cy, w, h).
    bboxes: tensor [N, 4] or list of shape [N, 4].
    """
    if isinstance(bboxes, list):
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
    
    x_min, y_min, x_max, y_max = bboxes.split(1, dim=1)
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    w = (x_max - x_min)
    h = (y_max - y_min)
    return torch.cat([cx, cy, w, h], dim=1)

def center_to_corner(bboxes):
    """
    Convert bounding boxes from center-size format (cx, cy, w, h)
    to corner format (x_min, y_min, x_max, y_max).
    bboxes: tensor [N, 4] or list of shape [N, 4].
    """
    if isinstance(bboxes, list):
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
    
    cx, cy, w, h = bboxes.split(1, dim=1)
    x_min = cx - w / 2.0
    x_max = cx + w / 2.0
    y_min = cy - h / 2.0
    y_max = cy + h / 2.0
    return torch.cat([x_min, y_min, x_max, y_max], dim=1)

def clamp_bbox_centerwh(bboxes):
    """
    Clamp bounding boxes in (cx, cy, w, h) format to [0,1].
    Useful if training in normalized coords to avoid out-of-range predictions.
    bboxes: shape (N, 4).
    """
    if isinstance(bboxes, list):
        bboxes = torch.tensor(bboxes, dtype=torch.float32)

    bboxes_clamped = torch.zeros_like(bboxes)
    # clamp cx, cy in [0,1]
    bboxes_clamped[:, 0] = torch.clamp(bboxes[:, 0], 0.0, 1.0)
    bboxes_clamped[:, 1] = torch.clamp(bboxes[:, 1], 0.0, 1.0)
    # clamp w, h in [0,1]
    bboxes_clamped[:, 2] = torch.clamp(bboxes[:, 2], 0.0, 1.0)
    bboxes_clamped[:, 3] = torch.clamp(bboxes[:, 3], 0.0, 1.0)
    return bboxes_clamped

def iou_func(bbox1, bbox2):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.
    bbox1, bbox2: tensors of shape (N, 4) in corner format (x_min, y_min, x_max, y_max).
    """
    bbox1_corner = center_to_corner(bbox1)
    bbox2_corner = center_to_corner(bbox2)
    iou = ops.box_iou(bbox1_corner, bbox2_corner).diagonal().mean().item()
    return iou

# ---------------------------------------------------------------------
# Dataset Processing
# ---------------------------------------------------------------------
class processData(Dataset):
    def __init__(self, image_folder, feature_folder, aux_folder, transform):
        self.image_folder = image_folder
        self.feature_folder = feature_folder
        self.aux_folder = aux_folder
        self.transform = transform

        self.image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])
        self.label_files = sorted([f for f in os.listdir(feature_folder) if f.endswith('.geojson')])
        self.aux_files = sorted([f for f in os.listdir(aux_folder) if f.endswith('.aux.xml')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_files[idx])
        label_path = os.path.join(self.feature_folder, self.label_files[idx])
        aux_path = os.path.join(self.aux_folder, self.aux_files[idx])

        # Parse GeoTransform and GeoJSON
        geo_transform = parse_geo_transform(aux_path)
        coords, _ = read_geojson(label_path)

        # Load and preprocess image
        image = Image.open(img_path).convert('RGB')
        width, height = image.size

        # Convert GeoJSON to pixel bounding box
        pixel_bbox = geojson_to_pixel_bboxes(coords, geo_transform, width, height)

        x_min, y_min, x_max, y_max = pixel_bbox
        x_min, x_max = x_min / width, x_max / width
        y_min, y_max = y_min / height, y_max / height
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        w = (x_max - x_min)
        h = (y_max - y_min)
        centered_bbox = [cx, cy, w, h]

        processed_image = self.transform(image)
        processed_bbox = torch.tensor(centered_bbox, dtype=torch.float32)

        return processed_image, processed_bbox
    
# ---------------------------------------------------------------------
#  CNN Model
# ---------------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Load a pretrained ResNet-18 model
        self.backbone = models.resnet18(pretrained=True)
        # self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 4)
        
        # Optionally freeze all layers
        for param in self.backbone.parameters():
            param.requires_grad = False

         # Optionally unfreeze `layer4`
        for child in self.backbone.layer4.named_children():
            for param in child[1].parameters():
                param.requires_grad = True

        # Optionally unfreeze the fully connected layer
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

        # Replace the final fully connected layer to match `output_features`
        self.backbone.fc = nn.Linear(in_features=512, out_features=4, bias=True)

    def forward(self, x):
        return torch.sigmoid(self.backbone(x))

# ---------------------------------------------------------------------
#  Model Training and Validation
# ---------------------------------------------------------------------
def train_model(model, train_loader, optimizer, criterion, num_epochs, device, epoch):
    model.train()
    epoch_loss = 0.0
    epoch_iou = 0.0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch +1}/{num_epochs} Training")
    for images, targets in progress_bar:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad() # Zero out gradients
        outputs = model(images)  # Forward Propagation
        loss = criterion(outputs, targets) # Compute loss
        loss.backward() # Backward Propagation
        optimizer.step() # Update weights
        epoch_loss += loss.item() # Update epoch loss
        iou = iou_func(outputs, targets) # Compute IoU
        epoch_iou += iou # Update epoch IoU
        num_batches += 1 # Update number of batches
        progress_bar.set_postfix(loss=loss.item(), IoU=iou) # Update progress bar

    # Average over the epoch
    avg_loss = epoch_loss / num_batches
    avg_iou = epoch_iou / num_batches
    return avg_loss, avg_iou, model

def validate_model(model, val_loader, criterion, num_epochs, device, epoch):
    model.eval()
    epoch_loss = 0.0
    epoch_iou = 0.0
    num_batches = 0

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc=f"Epoch {epoch +1}/{num_epochs} Validating")
        for images, targets in progress_bar:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)  # Bounding box predictions
            loss = criterion(outputs, targets) # Compute loss
            epoch_loss += loss.item() # Update epoch loss
            iou = iou_func(outputs, targets) # Compute IoU
            epoch_iou += iou # Update epoch IoU
            num_batches += 1 # Update number of batches
            progress_bar.set_postfix(loss=loss.item(), IoU=iou) # Update progress bar

    # Average over the epoch
    avg_loss = epoch_loss / num_batches
    avg_iou = epoch_iou / num_batches
    return avg_loss, avg_iou, model

# ---------------------------------------------------------------------
#  Resluts and Analysis
# --------------------------------------------------------------------- 
def generate_predictions(model, val_dataset, device, output_folder, num_images=16):
    """
    Picks 16 random images from val_dataset, predicts their bounding boxes,
    draws them along with true bounding boxes, and saves the results in a grid format.
    """
    os.makedirs(output_folder, exist_ok=True)
    model.eval()

    mean = torch.tensor([0.5, 0.5, 0.5])
    std = torch.tensor([0.5, 0.5, 0.5])

    grid_size = int(num_images ** 0.5)
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(12, 12))

    with torch.no_grad():
        for idx in range(num_images):
            random_idx = random.randint(0, len(val_dataset) - 1)
            image, true_bbox = val_dataset[random_idx]  # Now only returns image and true_bbox

            # Prepare inputs
            image_input = image.unsqueeze(0).to(device)  # Add batch dimension

            # Get predictions
            pred_bbox = model(image_input)[0].cpu()  # Predicted bounding box in (cx, cy, w, h)
            pred_bbox = clamp_bbox_centerwh(pred_bbox.unsqueeze(0))[0]  # Clamp predictions to [0,1]
            pred_corner = center_to_corner(pred_bbox.unsqueeze(0))[0]  # Convert to (x_min, y_min, x_max, y_max)

            # Denormalize the image for visualization
            image_denorm = image * std[:, None, None] + mean[:, None, None]
            pil_image = transforms.ToPILImage()(image_denorm).convert('RGB')

            width, height = pil_image.size
            x_min, y_min, x_max, y_max = pred_corner
            x_min, x_max = x_min * width, x_max * width
            y_min, y_max = y_min * height, y_max * height

            true_corner = center_to_corner(true_bbox.unsqueeze(0))[0]
            x_min_t, y_min_t, x_max_t, y_max_t = true_corner
            x_min_t, x_max_t = x_min_t * width, x_max_t * width
            y_min_t, y_max_t = y_min_t * height, y_max_t * height

            draw = ImageDraw.Draw(pil_image)
            draw.rectangle([x_min, y_min, x_max, y_max], outline='red', width=3)
            draw.rectangle([x_min_t, y_min_t, x_max_t, y_max_t], outline='green', width=3)

            ax = axs[idx // grid_size, idx % grid_size]
            ax.imshow(pil_image)
            ax.axis('off')
            ax.set_title(f"Sample {random_idx}")

    grid_path = os.path.join(output_folder, "random_predictions_grid.png")
    plt.tight_layout()
    plt.savefig(grid_path)
    plt.show()

def plot_training(train_losses, val_losses, train_ious, val_ious, output_folder):
    """
    Plots and saves loss and IoU metrics over epochs.
    """
    os.makedirs(output_folder, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)
    epochs = range(1, len(train_losses) + 1)

    # Plot Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss', marker='o', linestyle='-', markersize=5)
    plt.plot(epochs, val_losses, label='Validation Loss', marker='x', linestyle='-', markersize=5)
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, 'loss_curve.png'))
    plt.show()

    # Plot IoU
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_ious, label='Train IoU', marker='o', linestyle='-', markersize=5)
    plt.plot(epochs, val_ious, label='Validation IoU', marker='x', linestyle='-', markersize=5)
    plt.title('IoU Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, 'iou_curve.png'))
    plt.show()

# ---------------------------------------------------------------------
#  Main Function
# ---------------------------------------------------------------------
def main(lr=0.001, batch_size=16, num_epochs=3, pixel_size=512, device='cuda', reduce_data=False):
    """
    Main function to train and validate the CNN model.
    :param lr: Learning rate
    :param batch_size: Batch size
    :param num_epochs: Number of epochs
    :param pixel_size: Size of the input images after resizing
    :param device: Device to use for training (e.g., 'cpu' or 'cuda')
    :param loss_fn: Loss function to use ('SmoothL1_loss', 'iou_loss', 'giou_loss', 'combined_loss')
    :param reduce_data: Reduce data to a smaller subset for faster training
    """
    
    #  Data Processing
    # ---------------------------------------------------------------------
    train_image_folder = r"./data/raw/train/PS-RGB_tiled"
    train_feature_folder = r"./data/raw/train/geojson_aircraft_tiled"
    train_aux_folder = r"./data/raw/train/PS-RGB_tiled"
    test_image_folder = r"./data/raw/test/PS-RGB_tiled"
    test_feature_folder = r"./data/raw/test/geojson_aircraft_tiled"
    test_aux_folder = r"./data/raw/test/PS-RGB_tiled"

    # Training transformation with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((pixel_size, pixel_size)),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(15),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Test transformation (no augmentation)
    test_transform = transforms.Compose([
        transforms.Resize((pixel_size, pixel_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets and loaders
    train_dataset = processData(train_image_folder, train_feature_folder, train_aux_folder, train_transform)
    test_dataset = processData(test_image_folder, test_feature_folder, test_aux_folder, test_transform)

    # Reduce data for faster training
    if reduce_data:
        train_size = int(0.25 * len(train_dataset))
        test_size = int(0.10 * len(test_dataset))
        train_dataset, _ = torch.utils.data.random_split(train_dataset, [train_size, len(train_dataset) - train_size])
        test_dataset, _ = torch.utils.data.random_split(test_dataset, [test_size, len(test_dataset) - test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    #  Model Initialization
    # ---------------------------------------------------------------------
    if not torch.cuda.is_available() and device == 'cuda': # Check if CUDA is available
        print("CUDA is not available. Switching to CPU.")
        device = 'cpu'

    model = CNN()
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr) # Stochastic Gradient Descent
    criterion = nn.MSELoss()  # Mean Squared Error Loss
    # print(model)

    #  Training Loop
    # ---------------------------------------------------------------------
    training_losses = []
    training_ious = []
    validation_losses = []
    validation_ious = []

    for epoch in range(num_epochs):
        
        train_loss, train_iou, _ = train_model(model, train_loader, optimizer, criterion, num_epochs, device, epoch) # Train the model
        val_loss, val_iou, _ = validate_model(model, test_loader, criterion, num_epochs, device, epoch) # Validate the model

        # Print training and validation metrics
        print(f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation IoU: {val_iou:.4f}")

        # Save training metrics
        training_losses.append(train_loss)
        training_ious.append(train_iou)
        validation_losses.append(val_loss)
        validation_ious.append(val_iou)
        
        epoch += 1

    #  Results and Analysis
    # ---------------------------------------------------------------------
    timestamp = datetime.now().strftime("%H_%M__%d_%m_%y")  # Format: HH_MM__DD_MM_YY
    output_folder = os.path.join(r"./results/CNN", timestamp)
    os.makedirs(output_folder, exist_ok=True) # Create a unique folder for each run
    
    # Save training parameters to a file
    param_file = os.path.join(output_folder, 'parameters.txt')
    with open(param_file, 'w') as f:
        f.write(f"Learning Rate: {lr}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Num Epochs: {num_epochs}\n")
        f.write(f"Pixel Size: {pixel_size}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Loss Function: Mean Squared Error Loss\n")
        f.write(f"Reduce Data: {reduce_data}\n")

    # Plot training progress
    plot_training(training_losses, validation_losses, training_ious, validation_ious, output_folder)

    # Generate predictions
    generate_predictions(model, test_dataset, device=device, output_folder=output_folder)

    # Save model
    model_path = os.path.join(output_folder, 'cnn_model.pth')
    torch.save(model.state_dict(), model_path)

    # Evaluate IoU
    print("Training completed!")

if __name__ == "__main__":
   main(lr=1e-3, batch_size=64, num_epochs=100, pixel_size=512, device='cuda', reduce_data=False)
