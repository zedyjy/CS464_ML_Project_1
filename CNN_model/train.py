import os
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from torchvision import transforms
from torchvision.models import resnet18
from torchvision.ops import box_iou
from torchvision.ops import generalized_box_iou
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

# ---------------------------------------------------------------------
#  Loss Functions
# ---------------------------------------------------------------------
def iou_loss(preds, targets):
    preds_corner = center_to_corner(preds)
    targets_corner = center_to_corner(targets)
    iou = box_iou(preds_corner, targets_corner)
    if iou.numel() == 0:  # Avoid division by zero
        return torch.tensor(1.0, requires_grad=True)
    return 1 - iou.diagonal().mean()

def giou_loss(preds, targets):
    preds_corner = center_to_corner(preds)
    targets_corner = center_to_corner(targets)
    giou = generalized_box_iou(preds_corner, targets_corner)
    if giou.numel() == 0:  # Avoid division by zero
        return torch.tensor(1.0, requires_grad=True)
    return 1 - giou.diagonal().mean()

def combined_loss(preds, targets, alpha=0.5):
    iou = iou_loss(preds, targets)
    regression_loss = nn.SmoothL1Loss()(preds, targets)
    return alpha * iou + (1 - alpha) * regression_loss

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
        self.backbone = resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 4)

    def forward(self, x):
        return self.backbone(x)
    
    # def __init__(self, input_size):
    #     """
    #     CNN model for bounding box prediction from images.
    #     Automatically computes the size of the fully connected layer.
    #     :param input_size: The size of the input image (e.g., 512 for 512x512 images).
    #     """
    #     super(CNN, self).__init__()
    #     self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
    #     self.bn1 = nn.BatchNorm2d(32)
    #     self.relu1 = nn.ReLU()
    #     self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

    #     self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
    #     self.bn2 = nn.BatchNorm2d(64)
    #     self.relu2 = nn.ReLU()
    #     self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

    #     self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
    #     self.bn3 = nn.BatchNorm2d(128)
    #     self.relu3 = nn.ReLU()
    #     self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

    #     # Calculate flat_dim dynamically based on input_size and number of pooling layers
    #     final_size = input_size // (2 ** 3)  # 3 pooling layers
    #     self.flat_dim = final_size * final_size * 128

    #     # Fully connected layer for bounding box prediction
    #     self.fc_final = nn.Linear(self.flat_dim, 4)

    # def forward(self, x):
    #     """
    #     :param x: image tensor (B, 3, input_size, input_size)
    #     :return: bounding box (B, 4) => (cx, cy, w, h)
    #     """
    #     # CNN for image
    #     x = self.pool1(self.relu1(self.conv1(x)))
    #     x = self.pool2(self.relu2(self.conv2(x)))
    #     x = self.pool3(self.relu3(self.conv3(x)))
    #     x = x.view(x.size(0), -1)  # Flatten maintaining batch size

    #     # Predict bounding box
    #     out = self.fc_final(x)  # shape (B, 4)
    #     return out
    

# ---------------------------------------------------------------------
#  Model Training and Evaluation
# ---------------------------------------------------------------------
def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0.0
    epoch_iou = 0.0
    num_batches = 0

    progress_bar = tqdm(train_loader, desc="Training")
    for images, targets in progress_bar:
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)  # Bounding box predictions

        # Compute loss
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Compute IoU
        preds_corner = center_to_corner(clamp_bbox_centerwh(outputs))
        targets_corner = center_to_corner(targets)
        iou = box_iou(preds_corner, targets_corner).diagonal().mean().item()

        # Update epoch metrics
        epoch_loss += loss.item()
        epoch_iou += iou
        num_batches += 1

        # Update progress bar
        progress_bar.set_postfix(loss=loss.item(), iou=iou)

    # Average over the epoch
    avg_loss = epoch_loss / num_batches
    avg_iou = epoch_iou / num_batches
    return avg_loss, avg_iou

def validate_model(model, val_loader, criterion, device):
    model.eval()
    epoch_loss = 0.0
    epoch_iou = 0.0
    num_batches = 0

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation")
        for images, targets in progress_bar:
            images, targets = images.to(device), targets.to(device)

            outputs = model(images)  # Bounding box predictions

            # Compute loss
            loss = criterion(outputs, targets)

            # Compute IoU
            preds_corner = center_to_corner(clamp_bbox_centerwh(outputs))
            targets_corner = center_to_corner(targets)
            iou = box_iou(preds_corner, targets_corner).diagonal().mean().item()

            # Update epoch metrics
            epoch_loss += loss.item()
            epoch_iou += iou
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix(loss=loss.item(), iou=iou)

    # Average over the epoch
    avg_loss = epoch_loss / num_batches
    avg_iou = epoch_iou / num_batches
    return avg_loss, avg_iou

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

def evaluate_iou(model, val_loader, device):
    """
    Compute average IoU across the validation set for single bounding-box predictions.
    The target bounding boxes are in (cx, cy, w, h) [normalized].
    """
    model.eval()
    total_iou = 0.0
    count = 0

    with torch.no_grad():
        for images, targets in val_loader:  # Now only unpack images and targets
            images = images.to(device)
            targets = targets.to(device)

            preds = model(images)  # shape: (batch_size, 4)
            preds = clamp_bbox_centerwh(preds)  # Clamp to [0, 1]
            preds_corner = center_to_corner(preds)  # Convert to corner format
            targets_corner = center_to_corner(targets)  # Convert targets to corner format

            # Compute IoU for the batch
            iou_values = box_iou(preds_corner, targets_corner)  # IoU matrix
            batch_iou = iou_values.diagonal().mean().item()  # Mean IoU for the batch
            total_iou += batch_iou * images.size(0)  # Sum IoUs weighted by batch size
            count += images.size(0)

    avg_iou = total_iou / count if count > 0 else 0.0
    return avg_iou

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
def main(lr=0.001, batch_size=16, num_epochs=3, pixel_size=512, device='cpu', 
         loss_fn='giou_loss', reduce_data=False):
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
    # Paths
    train_image_folder = r"./data/raw/train/PS-RGB_tiled"
    train_feature_folder = r"./data/raw/train/geojson_aircraft_tiled"
    train_aux_folder = r"./data/raw/train/PS-RGB_tiled"
    test_image_folder = r"./data/raw/test/PS-RGB_tiled"
    test_feature_folder = r"./data/raw/test/geojson_aircraft_tiled"
    test_aux_folder = r"./data/raw/test/PS-RGB_tiled"

    # Create a unique folder for each run
    timestamp = datetime.now().strftime("%H:%M__%d.%m.%y")  # Format: hour:min__day.month.year
    output_folder = os.path.join(r"./results/CNN", timestamp)  # Combine base folder with timestamp
    os.makedirs(output_folder, exist_ok=True)
    
    # Save training parameters to a file
    param_file = os.path.join(output_folder, 'parameters.txt')
    with open(param_file, 'w') as f:
        f.write(f"Learning Rate: {lr}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Num Epochs: {num_epochs}\n")
        f.write(f"Pixel Size: {pixel_size}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Loss Function: {loss_fn}\n")
        f.write(f"Reduce Data: {reduce_data}\n")


    # Training transformation with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((pixel_size, pixel_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Test transformation (no augmentation)
    test_transform = transforms.Compose([
        transforms.Resize((pixel_size, pixel_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Create datasets and loaders
    if reduce_data:
        train_size = int(0.25 * len(train_dataset))
        test_size = int(0.10 * len(test_dataset))
        train_dataset, _ = torch.utils.data.random_split(train_dataset, [train_size, len(train_dataset) - train_size])
        test_dataset, _ = torch.utils.data.random_split(test_dataset, [test_size, len(test_dataset) - test_size])
    else:
        train_dataset = processData(train_image_folder, train_feature_folder, train_aux_folder, train_transform)
        test_dataset = processData(test_image_folder, test_feature_folder, test_aux_folder, test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model, optimizer, and loss
    model = CNN()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)  # Adjust T_max based on epochs
    
    # Select loss function
    if loss_fn == 'giou_loss':
        criterion = giou_loss
    elif loss_fn == 'iou_loss':
        criterion = iou_loss
    elif loss_fn == 'combined_loss':
        criterion = combined_loss
    elif loss_fn == 'SmoothL1_loss':
        criterion = nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_fn}")

    train_losses = []
    train_ious = []
    val_losses = []
    val_ious = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Train
        avg_train_loss, avg_train_iou = train_model(model, train_loader, optimizer, criterion, device)
        train_losses.append(avg_train_loss)
        train_ious.append(avg_train_iou)

        # Validate
        avg_val_loss, avg_val_iou = validate_model(model, test_loader, criterion, device)
        val_losses.append(avg_val_loss)
        val_ious.append(avg_val_iou)

        # Update scheduler
        scheduler.step(avg_val_loss)  # Update learning rate based on the latest validation loss

        # Print epoch results
        print(f"Train Loss: {avg_train_loss:.4f}, Train IoU: {avg_train_iou:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Val IoU: {avg_val_iou:.4f}")

    # Plot training progress
    plot_training(train_losses, val_losses, train_ious, val_ious, output_folder)

    # Generate predictions
    generate_predictions(model, test_dataset, device=device, output_folder=output_folder)

    # Save model
    model_path = os.path.join(output_folder, 'cnn_model.pth')
    torch.save(model.state_dict(), model_path)

    # Evaluate IoU
    print("Evaluating average IoU on the validation set...")
    avg_iou = evaluate_iou(model, test_loader, device=device)
    print(f"Average Validation IoU: {avg_iou:.4f}")
    print("Training completed!")

if __name__ == "__main__":
    main(lr=0.001, batch_size=8, num_epochs=20, pixel_size=256, device='cpu')
