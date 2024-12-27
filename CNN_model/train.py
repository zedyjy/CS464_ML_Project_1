import os
import json
import random
import logging
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from torchvision import models, ops, transforms
from torchvision.models import ResNet18_Weights
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
        Args:
            aux_path (str): The file path to the .aux.xml file containing the GeoTransform metadata.

        Returns:
            list of float: A list of six floating-point numbers representing the GeoTransform.

        Raises:
            FileNotFoundError: If the specified .aux.xml file does not exist.
            ET.ParseError: If there is an error parsing the XML file.
            AttributeError: If the GeoTransform element is not found in the XML file.
    """
    logging.debug(f"Parsing GeoTransform from: {aux_path}")
    tree = ET.parse(aux_path)
    root = tree.getroot()
    geo_transform = root.find("GeoTransform").text.strip().split(",")
    geo_transform = [float(value) for value in geo_transform]
    logging.debug(f"GeoTransform parsed: {geo_transform}")
    logging.debug( "-------------------------------------------------")
    return geo_transform

def read_geojson(geojson_path):
    """
    Read and parse a GeoJSON file to extract coordinates and features.
    Parameters:
    geojson_path (str): The file path to the GeoJSON file.
    Returns:
    tuple: A tuple containing:
        - coords (list): A list of coordinates extracted from the first feature's geometry.
        - features (list): A list of features extracted from the first feature's properties, including:
            - length (float): Length property of the feature.
            - wingspan (float): Wingspan property of the feature.
            - area (float): Area property of the feature.
            - wing_type_code (int): Encoded wing type (0: straight, 1: swept, 2: other).
            - wing_position_code (int): Encoded wing position (0: high mounted, 1: low/mid mounted, 2: other).
            - canard (int): Presence of canards (1: yes, 0: no).
            - num_engines (int): Number of engines.
            - num_tailfins (int): Number of tail fins.
            - faa_class (int): FAA wingspan class.
    Raises:
    ValueError: If the coordinates in the GeoJSON are invalid.
    """
    logging.debug(f"Reading GeoJSON from: {geojson_path}")
    with open(geojson_path, 'r') as f:
        data = json.load(f)
        
    if not data['features']:
        logging.warning("No features found in GeoJSON. Using dummy coordinates.")    
        coords = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]  # Dummy coordinates
        features = [0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0]
    else:
        first_feature = data['features'][0]
        coords = first_feature['geometry']['coordinates']
        logging.debug(f"First feature coordinates: {coords}")
        props = first_feature.get('properties', {})
        logging.debug(f"Feature properties: {props}")
        
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
            coords = list(zip(xs, ys))
        
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

        canard = 1 if props.get('canards', 'yes') == 'yes' else 0
        num_engines = props.get('num_engines', 0)
        num_tailfins = props.get('num_tail_fins', 0)
        faa_class = props.get('faa_wingspan_class', 0)

        features = [length, wingspan, area, wing_type_code, wing_position_code, canard, num_engines, num_tailfins, faa_class]
    
    logging.debug(f"GeoJSON Coords: {coords}")
    logging.debug(f"GeoJSON Features: {features}")
    logging.debug( "-------------------------------------------------")
    return coords, features

def geo_to_pixel(lon, lat, geo_transform, image_width, image_height):
    """
    Convert geographic coordinates (longitude, latitude) to image pixel coordinates.
    Parameters:
        lon (float): Longitude of the geographic coordinate.
        lat (float): Latitude of the geographic coordinate.
        geo_transform (tuple): A tuple containing the affine transformation coefficients.
        image_width (int): Width of the image in pixels.
        image_height (int): Height of the image in pixels.
    Returns:
        tuple: A tuple containing the x and y pixel coordinates.
    Notes:
        The function clamps the pixel coordinates to ensure they fall within the image dimensions.
    """
    logging.debug(f"Converting geo-coordinates to pixel coordinates:")
    logging.debug(f"Inputs - lon: {lon}, lat: {lat}, geo_transform: {geo_transform}, "
                  f"image_width: {image_width}, image_height: {image_height}")

    # Parse geotransformation coefficients
    x_origin, pixel_width, _, y_origin, _, pixel_height = geo_transform
    logging.debug(f"Parsed GeoTransform - x_origin: {x_origin}, pixel_width: {pixel_width}, "
                  f"y_origin: {y_origin}, pixel_height: {pixel_height}")

    # Compute pixel coordinates
    x_pixel = int((lon - x_origin) / pixel_width)
    y_pixel = int((y_origin - lat) / abs(pixel_height))
    logging.debug(f"Computed pixel coordinates before clamping - x_pixel: {x_pixel}, y_pixel: {y_pixel}")

    # Clamp pixel coordinates to image boundaries
    x_pixel_clamped = max(0, min(x_pixel, image_width - 1))
    y_pixel_clamped = max(0, min(y_pixel, image_height - 1))
    logging.debug(f"Clamped pixel coordinates - x_pixel: {x_pixel_clamped}, y_pixel: {y_pixel_clamped}")
    return x_pixel_clamped, y_pixel_clamped

def geojson_to_pixel_bboxes(coords, geo_transform, image_width, image_height):
    """
    Convert GeoJSON bounding boxes to image pixel bounding boxes.    
    Args:
        coords (list of tuples): List of (longitude, latitude) tuples representing the GeoJSON coordinates.
        geo_transform (tuple): Geotransformation parameters for converting geo-coordinates to pixel coordinates.
        image_width (int): Width of the image in pixels.
        image_height (int): Height of the image in pixels.
    Returns:
        tensor: A tensor of shape [4] containing (x_min, y_min, x_max, y_max) which are the pixel coordinates of the bounding box.
    """
    logging.debug("------------------- Converting GeoJSON to Pixel Bounding Box -------------------")
    logging.debug(f"Input Coordinates: {coords}")
    logging.debug(f"GeoTransform: {geo_transform}")
    logging.debug(f"Image Dimensions: width={image_width}, height={image_height}")
    
    pixel_coords = [geo_to_pixel(lon, lat, geo_transform, image_width, image_height) for lon, lat in coords]
    logging.debug(f"Pixel Coordinates: {pixel_coords}")

    x_coords = [p[0] for p in pixel_coords]
    y_coords = [p[1] for p in pixel_coords]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    bboxes = torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)
    logging.debug(f"Pixel Bounding Box Tensor: {bboxes}")
    return bboxes

def norm_bbox(bboxes, image_width, image_height):
    """
    Normalize bounding boxes to the range [0, 1].
    
    Args:
        bboxes (tensor): Tensor of shape [N, 4] in corner format (x_min, y_min, x_max, y_max).
        image_width (int): Width of the image.
        image_height (int): Height of the image.
        
    Returns:
        tensor: Tensor of shape [N, 4] with normalized bounding boxes.
    """
    logging.debug("-------------------------- Normalizing Bounding Boxes --------------------------")
    logging.debug(f"Input Bounding Boxes: {bboxes}")

    # Ensure bboxes has the correct shape
    if bboxes.ndimension() == 1 and bboxes.size(0) == 4:
        bboxes = bboxes.unsqueeze(0)
    elif bboxes.ndimension() != 2 or bboxes.size(1) != 4:
        raise ValueError(f"Expected bboxes to have shape [N, 4], but got {bboxes.shape}")

    # Normalize bounding box coordinates
    x_min, y_min, x_max, y_max = bboxes.split(1, dim=1) 
    x_min = x_min / image_width
    y_min = y_min / image_height
    x_max = x_max / image_width
    y_max = y_max / image_height

    # Clamp to ensure values are in [0, 1]
    x_min = torch.clamp(x_min, 0, 1)
    y_min = torch.clamp(y_min, 0, 1)
    x_max = torch.clamp(x_max, 0, 1)
    y_max = torch.clamp(y_max, 0, 1)

    # Combine back into tensor
    result = torch.cat([x_min, y_min, x_max, y_max], dim=1)

    logging.debug(f"Normalized Bounding Boxes: {result}")
    return result

def corner_to_center(bboxes):
    """
    Convert bounding boxes from corner format (x_min, y_min, x_max, y_max)
    to center-size format (cx, cy, w, h).
    Args:
        bboxes (tensor or list): Tensor of shape [N, 4] or list of shape [N, 4].
    Returns:
        tensor: Tensor of shape [N, 4] in format (cx, cy, w, h).
    """
    logging.debug("--------------------- Converting Format of Bounding Boxes ----------------------")
    logging.debug(f"Input Bounding Boxes (Corner Format): {bboxes}")

    # Ensure bboxes has the correct shape
    if bboxes.ndimension() == 1 and bboxes.size(0) == 4:
        bboxes = bboxes.unsqueeze(0)  # Add batch dimension
    elif bboxes.ndimension() != 2 or bboxes.size(1) != 4:
        raise ValueError(f"Expected bboxes to have shape [N, 4], but got {bboxes.shape}")
    
    # Calculate center coordinates and dimensions
    x_min, y_min, x_max, y_max = bboxes.split(1, dim=1)
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    w = x_max - x_min
    h = y_max - y_min

    # Concatenate results into a single tensor
    result = torch.cat([cx, cy, w, h], dim=1)
    logging.debug(f"Converted Bounding Boxes (Center Format): {result}")
    return result

def center_to_corner(bboxes):
    """
    Convert bounding boxes from center-size format (cx, cy, w, h)
    to corner format (x_min, y_min, x_max, y_max).
    Args:
        bboxes (tensor or list): Tensor of shape [N, 4] or list of shape [N, 4].
    Returns:
        tensor: Tensor of shape [N, 4] in format (x_min, y_min, x_max, y_max).
    """
    logging.debug("--------------------- Converting Format of Bounding Boxes ----------------------")   
    logging.debug(f"Input Bounding Boxes (Center Format): {bboxes}")

    # Split into cx, cy, w, h
    cx, cy, w, h = bboxes.split(1, dim=1)
    logging.debug(f"cx: {cx.squeeze().tolist()}, cy: {cy.squeeze().tolist()}, "
                  f"w: {w.squeeze().tolist()}, h: {h.squeeze().tolist()}")

    # Calculate corner coordinates
    x_min = cx - w / 2.0
    x_max = cx + w / 2.0
    y_min = cy - h / 2.0
    y_max = cy + h / 2.0
    result = torch.cat([x_min, y_min, x_max, y_max], dim=1)

    logging.debug(f"Converted Bounding Boxes (Corner Format): {result}")
    return result

def iou_func(bbox1, bbox2):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        bbox1 (tensor): Tensor of shape (N, 4) in center format (cx, cy, w, h).
        bbox2 (tensor): Tensor of shape (N, 4) in center format (cx, cy, w, h).
    
    Returns:
        float: The average IoU across all pairs of bounding boxes.
    """
    logging.debug("--------------------------------- IoU Function ---------------------------------")
    logging.debug(f"Input bbox1 (Center Format): {bbox1}")
    logging.debug(f"Input bbox2 (Center Format): {bbox2}")

    # Convert bounding boxes from center format to corner format
    bbox1_corner = center_to_corner(bbox1)
    bbox2_corner = center_to_corner(bbox2)

    logging.debug(f"Converted bbox1 (Corner Format): {bbox1_corner}")
    logging.debug(f"Converted bbox2 (Corner Format): {bbox2_corner}")

    # Calculate IoU
    iou_matrix = ops.box_iou(bbox1_corner, bbox2_corner)
    logging.debug(f"IoU Matrix: {iou_matrix}")

    # Compute average IoU
    iou = iou_matrix.diagonal().mean().item()
    logging.debug(f"Average IoU: {iou}")
    return iou

# ---------------------------------------------------------------------
# Dataset Processing
# ---------------------------------------------------------------------
class processData(Dataset):
    def __init__(self, image_folder, feature_folder, aux_folder, transform):
        """
        Dataset class to load images, GeoJSON files, and auxiliary files.
        Args:
            image_folder (str): Path to the folder containing images.
            feature_folder (str): Path to the folder containing GeoJSON files.
            aux_folder (str): Path to the folder containing auxiliary files.
            transform (callable): Transformations to apply to the images.
        """
        self.image_folder = image_folder
        self.feature_folder = feature_folder
        self.aux_folder = aux_folder
        self.transform = transform

        self.image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])
        self.label_files = sorted([f for f in os.listdir(feature_folder) if f.endswith('.geojson')])
        self.aux_files = sorted([f for f in os.listdir(aux_folder) if f.endswith('.aux.xml')])

        # Log dataset initialization
        logging.debug(f"Initialized processData with {len(self.image_files)} images, "
                      f"{len(self.label_files)} GeoJSON files, and {len(self.aux_files)} auxiliary files.")

    def __len__(self):
        """
        Return the total number of items in the dataset.
        """
        length = len(self.image_files)
        logging.debug(f"Dataset length: {length}")
        return length

    def __getitem__(self, idx):
        """
        Fetch the item at the given index.
        Args:
            idx (int): Index of the item to fetch.
        Returns:
            processed_image (tensor): Transformed image.
            processed_bbox (tensor): Bounding box in (cx, cy, w, h) format.
        """
        # Paths for the image, label, and auxiliary files
        img_path = os.path.join(self.image_folder, self.image_files[idx])
        label_path = os.path.join(self.feature_folder, self.label_files[idx])
        aux_path = os.path.join(self.aux_folder, self.aux_files[idx])

        # Log file paths
        logging.debug(f"Processing index {idx}:")
        logging.debug(f"Image path: {img_path}")
        logging.debug(f"Label path: {label_path}")
        logging.debug(f"Auxiliary path: {aux_path}")

        # Parse GeoTransform and GeoJSON
        geo_transform = parse_geo_transform(aux_path)
        coords, _ = read_geojson(label_path)
        logging.debug(f"Parsed GeoTransform: {geo_transform}")
        logging.debug(f"Parsed GeoJSON coordinates: {coords}")

        # Load and preprocess image
        image = Image.open(img_path).convert('RGB')
        width, height = image.size
        logging.debug(f"Loaded image size: {width}x{height}")

        # Convert GeoJSON to pixel bounding box
        pixel_bbox = geojson_to_pixel_bboxes(coords, geo_transform, width, height)

        # Normalize the bounding box to [0,1]
        normalized_bbox = norm_bbox(pixel_bbox, width, height)
        
        # Convert pixel bounding box to center format
        centered_bbox = corner_to_center(normalized_bbox)
        
        # Apply transformations to the image
        processed_image = self.transform(image)
        processed_bbox = centered_bbox.clone().detach()
        processed_bbox = processed_bbox.squeeze(0)  # Ensure target tensor has shape [4]
        logging.debug(f"Processed image tensor shape: {processed_image.shape}")
        logging.debug(f"Processed bounding box tensor: {processed_bbox}")
        return processed_image, processed_bbox
    
# ---------------------------------------------------------------------
#  CNN Model
# ---------------------------------------------------------------------
class CNN(nn.Module):
    """
    A Convolutional Neural Network (CNN) model that uses a pretrained ResNet-18 backbone.
    Attributes:
        backbone (torchvision.models.ResNet): The ResNet-18 model used as the backbone of the CNN.
    Methods:
        __init__(): Initializes the CNN model, loads a pretrained ResNet-18 model, optionally freezes layers,
                    and replaces the final fully connected layer to match the desired output features.
        forward(x): Defines the forward pass of the model, applying the backbone and a sigmoid activation function.
    Example:
        model = CNN()
        output = model(input_tensor)
    """
    def __init__(self):
        super(CNN, self).__init__()
        # Load a pretrained ResNet-18 model
        self.backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        for name, param in self.backbone.named_parameters():
            if "layer2" in name or "layer3" in name or "layer4" in name or "fc" in name:
                param.requires_grad = True

        # Replace the final fully connected layer to match `output_features`
        self.backbone.fc = nn.Linear(in_features=512, out_features=4, bias=True)

    def forward(self, x):
        return torch.sigmoid(self.backbone(x))
    
    def ciou_loss(self, pred, target):
        """
        Compute the Complete Intersection over Union (CIoU) loss between predicted and target bounding boxes.

        Args:
            pred (torch.Tensor): Predicted bounding boxes of shape (N, 4), where N is the number of boxes.
                                 Each box is represented by [x1, y1, x2, y2].
            target (torch.Tensor): Target bounding boxes of shape (N, 4), where N is the number of boxes.
                                   Each box is represented by [x1, y1, x2, y2].

        Returns:
            torch.Tensor: The CIoU loss value.

        """
        pred = torch.clamp(pred, 0, 1)
        target = torch.clamp(target, 0, 1)
        iou_matrix = ops.box_iou(pred, target)
        iou = iou_matrix.diagonal()

        # Center distance
        pred_center = (pred[:, :2] + pred[:, 2:]) / 2
        target_center = (target[:, :2] + target[:, 2:]) / 2
        center_distance = torch.sum((pred_center - target_center) ** 2, dim=1)

        # Diagonal length of enclosing box
        enclosing_min = torch.min(pred[:, :2], target[:, :2])
        enclosing_max = torch.max(pred[:, 2:], target[:, 2:])
        diagonal_length = torch.sum((enclosing_max - enclosing_min) ** 2, dim=1)

        # Aspect ratio consistency
        pred_wh = pred[:, 2:] - pred[:, :2]
        target_wh = target[:, 2:] - target[:, :2]
        v = (4 / (3.14159 ** 2)) * torch.pow(torch.atan(pred_wh[:, 0] / pred_wh[:, 1]) -
                                              torch.atan(target_wh[:, 0] / target_wh[:, 1]), 2)
        alpha = v / (1 - iou + v + 1e-7)

        ciou = iou - center_distance / diagonal_length - alpha * v
        return 1 - ciou.mean()

    def combined_loss(self, pred, target, alpha=0.3, beta=0.7):
        Smooth_L1_loss = nn.SmoothL1Loss()(pred, target)
        IoU_Loss = self.ciou_loss(pred, target)
        combined_loss = alpha * (Smooth_L1_loss) + beta * (IoU_Loss)
        return combined_loss

# ---------------------------------------------------------------------
#  Model Training and Validation
# ---------------------------------------------------------------------
def train_model(model, train_loader, optimizer, criterion, num_epochs, device, epoch):
    """
    Trains the given model for one epoch with added logging for debugging.
    Args:
        model (torch.nn.Module): The neural network model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        optimizer (torch.optim.Optimizer): Optimizer for updating the model parameters.
        criterion (torch.nn.Module): Loss function to be used.
        num_epochs (int): Total number of epochs for training.
        device (torch.device): Device to run the training on (e.g., 'cpu' or 'cuda').
        epoch (int): The current epoch number.
    Returns:
        tuple: A tuple containing:
            - avg_loss (float): The average loss over the epoch.
            - avg_iou (float): The average Intersection over Union (IoU) over the epoch.
            - model (torch.nn.Module): The trained model.
    """
    logging.debug(f"Starting training for epoch {epoch + 1}/{num_epochs}")
    model.train()
    epoch_loss = 0.0
    epoch_iou = 0.0
    num_batches = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Training")
    for batch_idx, (images, targets) in enumerate(progress_bar):
        logging.debug(f"Batch {batch_idx + 1}: Loading data")
        images, targets = images.to(device), targets.to(device)
        logging.debug(f"Batch {batch_idx + 1}: Data shapes - Images: {images.shape}, Targets: {targets.shape}")
        
        optimizer.zero_grad()  # Zero out gradients
        logging.debug(f"Batch {batch_idx + 1}: Forward pass")
        outputs = model(images)  # Forward Propagation
        logging.debug(f"Batch {batch_idx + 1}: Output shape - {outputs.shape}")
        
        loss = criterion(outputs, targets)  # Compute loss
        logging.debug(f"Batch {batch_idx + 1}: Loss computed - {loss.item():.6f}")
        
        loss.backward()  # Backward Propagation
        optimizer.step()  # Update weights
        
        epoch_loss += loss.item()  # Update epoch loss
        iou = iou_func(outputs, targets)  # Compute IoU
        
        epoch_iou += iou  # Update epoch IoU
        num_batches += 1  # Update number of batches
        
        progress_bar.set_postfix(loss=loss.item(), IoU=iou)  # Update progress bar

    # Average over the epoch
    avg_loss = epoch_loss / num_batches
    avg_iou = epoch_iou / num_batches
    logging.debug(f"Epoch {epoch + 1}: Average Loss - {avg_loss:.6f}, Average IoU - {avg_iou:.6f}")

    return avg_loss, avg_iou, model

def validate_model(model, val_loader, criterion, num_epochs, device, epoch):
    """
    Validates the given model for one epoch with added logging for debugging.
    Args:
        model (torch.nn.Module): The neural network model to be validated.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation data.
        criterion (torch.nn.Module): Loss function to be used.
        num_epochs (int): Total number of epochs for training.
        device (torch.device): Device to run the validation on (e.g., 'cpu' or 'cuda').
        epoch (int): The current epoch number.
    Returns:
        tuple: A tuple containing:
            - avg_loss (float): The average loss over the epoch.
            - avg_iou (float): The average Intersection over Union (IoU) over the epoch.
            - model (torch.nn.Module): The model after validation.
    """
    logging.debug(f"Starting validation for epoch {epoch + 1}/{num_epochs}")
    model.eval()
    epoch_loss = 0.0
    epoch_iou = 0.0
    num_batches = 0

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Validating")
        for batch_idx, (images, targets) in enumerate(progress_bar):
            logging.debug(f"Batch {batch_idx + 1}: Loading data")
            images, targets = images.to(device), targets.to(device)
            logging.debug(f"Batch {batch_idx + 1}: Data shapes - Images: {images.shape}, Targets: {targets.shape}")
            
            outputs = model(images)  # Bounding box predictions
            outputs = torch.clamp(outputs, 0, 1)

            logging.debug(f"Batch {batch_idx + 1}: Output shape - {outputs.shape}")
            
            loss = criterion(outputs, targets)  # Compute loss
            logging.debug(f"Batch {batch_idx + 1}: Loss computed - {loss.item():.6f}")
            
            epoch_loss += loss.item()  # Update epoch loss
            iou = iou_func(outputs, targets)  # Compute IoU
            logging.debug(f"Batch {batch_idx + 1}: IoU computed - {iou:.6f}")
            
            epoch_iou += iou  # Update epoch IoU
            num_batches += 1  # Update number of batches
            
            progress_bar.set_postfix(loss=loss.item(), IoU=iou)  # Update progress bar

    # Average over the epoch
    avg_loss = epoch_loss / num_batches
    avg_iou = epoch_iou / num_batches
    logging.debug(f"Epoch {epoch + 1}: Average Loss - {avg_loss:.6f}, Average IoU - {avg_iou:.6f}")

    return avg_loss, avg_iou, model

# ---------------------------------------------------------------------
#  Resluts and Analysis
# --------------------------------------------------------------------- 
def generate_predictions(model, val_dataset, device, output_folder, num_images=16):
    """
    Generate and save a grid of predictions from a model on a validation dataset.

    Args:
        model (torch.nn.Module): The trained model used for generating predictions.
        val_dataset (torch.utils.data.Dataset): The validation dataset containing images and true bounding boxes.
        device (torch.device): The device (CPU or GPU) to run the model on.
        output_folder (str): The folder where the output grid image will be saved.
        num_images (int, optional): The number of random images to generate predictions for. Default is 16.

    Returns:
        None
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
            pred_bbox = model(image_input)[0].to(device)  # Predicted bounding box in (cx, cy, w, h)
            # pred_bbox = clamp_bbox_centerwh(pred_bbox.unsqueeze(0))[0]  # Clamp predictions to [0,1]
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
    Plots training and validation loss and IoU over epochs and saves the plots to the specified output folder.

    Args:
        train_losses (list of float): List of training loss values for each epoch.
        val_losses (list of float): List of validation loss values for each epoch.
        train_ious (list of float): List of training IoU values for each epoch.
        val_ious (list of float): List of validation IoU values for each epoch.
        output_folder (str): Path to the folder where the plots will be saved.

    Returns:
        None
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
def main(lr=0.001, batch_size=16, num_epochs=3, pixel_size=512, reduce_data=False, patience=5):
    """
    Main function to train and validate a Convolutional Neural Network (CNN) model.
    Args:
        lr (float): Learning rate for the optimizer. Default is 0.001.
        batch_size (int): Number of samples per batch. Default is 16.
        num_epochs (int): Number of epochs for training. Default is 3.
        pixel_size (int): Size of the image pixels. Default is 512.
        reduce_data (bool): Flag to reduce the dataset size for faster training. Default is False.
        patience (int): Number of epochs to wait before early stopping. Default is 5.
    Returns:
        None
    """

    # Create output folder
    timestamp = datetime.now().strftime("%H_%M__%d_%m_%y")
    output_folder = os.path.join(r"./results/CNN", timestamp)
    os.makedirs(output_folder, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,  # Set to DEBUG for step-by-step tracing
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Log to console
            # Save logs to a file in the results folder
            logging.FileHandler(os.path.join(output_folder, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"))
        ]
    )  

    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")
    
    # Save parameters to a file
    param_file = os.path.join(output_folder, 'parameters.txt')
    with open(param_file, 'w') as f:
        f.write(f"Learning Rate: {lr}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Num Epochs: {num_epochs}\n")
        f.write(f"Pixel Size: {pixel_size}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Loss Function: Complete IoU + Smooth_L1 Combined Error Loss\n")
        f.write(f"Reduce Data: {reduce_data}\n")

    # Data Processing
    logging.debug("Initializing data processing")
    train_image_folder = r"./data/raw/train/PS-RGB_tiled"
    train_feature_folder = r"./data/raw/train/geojson_aircraft_tiled"
    train_aux_folder = r"./data/raw/train/PS-RGB_tiled"
    test_image_folder = r"./data/raw/test/PS-RGB_tiled"
    test_feature_folder = r"./data/raw/test/geojson_aircraft_tiled"
    test_aux_folder = r"./data/raw/test/PS-RGB_tiled"

    transform = transforms.Compose([
        transforms.Resize((pixel_size, pixel_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    logging.debug("Loading datasets")
    train_dataset = processData(train_image_folder, train_feature_folder, train_aux_folder, transform)
    test_dataset = processData(test_image_folder, test_feature_folder, test_aux_folder, transform)

    # Reduce data for faster training
    if reduce_data:
        logging.debug("Reducing dataset size")
        train_size = int(0.25 * len(train_dataset))
        test_size = int(0.10 * len(test_dataset))
        train_dataset, _ = torch.utils.data.random_split(train_dataset, [train_size, len(train_dataset) - train_size])
        test_dataset, _ = torch.utils.data.random_split(test_dataset, [test_size, len(test_dataset) - test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model Initialization
    logging.debug("Initializing model and optimizer")
    model = CNN()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = model.combined_loss

    # Training Loop
    logging.debug("Starting CNN training")
    training_losses = []
    training_ious = []
    validation_losses = []
    validation_ious = []

    # Early Stopping Variables
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        logging.debug(f"Starting epoch {epoch + 1}/{num_epochs}")
        train_loss, train_iou, _ = train_model(model, train_loader, optimizer, criterion, num_epochs, device, epoch)
        val_loss, val_iou, _ = validate_model(model, test_loader, criterion, num_epochs, device, epoch)
        
        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model_path = os.path.join(output_folder, 'cnn_model.pth')
            torch.save(model.state_dict(), model_path)
            logging.info(f"New best model saved at epoch {epoch + 1}")
        else:
            patience_counter += 1
            logging.info(f"Patience counter: {patience_counter}/{patience}")

        # Early stopping check
        if patience_counter >= patience: 
            logging.info("Early stopping triggered.")
            break

        logging.info(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}")
        logging.info(f"Epoch {epoch + 1}: Validation Loss: {val_loss:.4f}, Validation IoU: {val_iou:.4f}")

        training_losses.append(train_loss)
        training_ious.append(train_iou)
        validation_losses.append(val_loss)
        validation_ious.append(val_iou)

    # Results and Analysis
    logging.debug("Saving results and generating outputs")
    plot_training(training_losses, validation_losses, training_ious, validation_ious, output_folder)
    generate_predictions(model, test_dataset, device=device, output_folder=output_folder)

    logging.info("Training completed successfully")

if __name__ == "__main__":
    # Debug Setup
    # main(lr=1e-3, batch_size=4, num_epochs=1, pixel_size=256, reduce_data=True)
    
    # Production Setup
    main(lr=1e-3, batch_size=32, num_epochs=100, pixel_size=512, reduce_data=False)
