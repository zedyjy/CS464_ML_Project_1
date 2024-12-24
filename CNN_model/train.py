import os
import json
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
from torchvision.ops import box_iou
from PIL import Image, ImageDraw
from tqdm import tqdm

# ---------------------------------------------------------------------
#  Utility Functions
# ---------------------------------------------------------------------
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
#  CNN Model
# ---------------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self, extra_in=3):
        """
        :param extra_in: number of extra features (e.g. length, wingspan, wing_position_code).
        """
        super(CNN, self).__init__()
        # Convolutional backbone for images
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # After the third pool, if input is 256x256 => output size is 32x32 with 64 channels => 32 * 32 * 64
        self.flat_dim = 32 * 32 * 64

        # A small MLP for the extra features
        # You can make this bigger or smaller as you wish
        self.extra_fc = nn.Sequential(
            nn.Linear(extra_in, 16),
            nn.ReLU()
        )

        # Combine image features + extra features => final bounding box
        self.fc_final = nn.Linear(self.flat_dim + 16, 4)

    def forward(self, x, extras):
        """
        :param x: image tensor (B, 3, 512, 512)
        :param extras: extra feature tensor (B, extra_in)
        :return: bounding box (B, 4) => (cx, cy, w, h)
        """
        # CNN for image
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = x.view(-1, self.flat_dim)

        # MLP for extras
        e = self.extra_fc(extras)

        # Combine
        combined = torch.cat([x, e], dim=1)  # shape (B, flat_dim+16)
        out = self.fc_final(combined)        # shape (B, 4)
        return out

# ---------------------------------------------------------------------
#  Model Training and Evaluation
# ---------------------------------------------------------------------
def train_model(model, train_loader, optimizer, criterion, num_epochs=1, device='cpu'):
    model.to(device)
    batch_train_losses = []
    batch_train_ious = []

    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{num_epochs}")
        for images, extra_feats, targets in progress_bar:
            images, extra_feats, targets = images.to(device), extra_feats.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(images, extra_feats)  # Bounding box predictions

            # Compute loss
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Compute IoU
            preds_corner = center_to_corner(clamp_bbox_centerwh(outputs))
            targets_corner = center_to_corner(targets)
            iou = box_iou(preds_corner, targets_corner).diagonal().mean().item()

            batch_train_losses.append(loss.item())
            batch_train_ious.append(iou)

            # Update progress bar
            progress_bar.set_postfix(loss=loss.item(), iou=iou)

    return batch_train_losses, batch_train_ious

def validate_model(model, val_loader, criterion, device='cpu'):
    model.eval()
    batch_val_losses = []
    batch_val_ious = []

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation")
        for images, extra_feats, targets in progress_bar:
            images, extra_feats, targets = images.to(device), extra_feats.to(device), targets.to(device)

            outputs = model(images, extra_feats)  # Bounding box predictions

            # Compute loss
            loss = criterion(outputs, targets)
            batch_val_losses.append(loss.item())

            # Compute IoU
            preds_corner = center_to_corner(clamp_bbox_centerwh(outputs))
            targets_corner = center_to_corner(targets)
            iou = box_iou(preds_corner, targets_corner).diagonal().mean().item()
            batch_val_ious.append(iou)

            # Update progress bar
            progress_bar.set_postfix(loss=loss.item(), iou=iou)

    return batch_val_losses, batch_val_ious

# ---------------------------------------------------------------------
#  Resluts and Analysis
# ---------------------------------------------------------------------
def analyze_feature_importance(model, feature_names, output_folder='./results/CNN'):
    """
    Analyzes the importance of each input feature by looking at the model's learned weights.
    Produces and saves a bar plot of feature importance.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Extract learned weights for the extra features
    feature_weights = model.extra_fc[0].weight.abs().mean(dim=0).detach().cpu().numpy()

    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, feature_weights, color='steelblue')
    plt.xlabel("Average Weight Magnitude")
    plt.ylabel("Feature")
    plt.title("Feature Importance")
    plt.grid(axis='x')

    plot_path = os.path.join(output_folder, 'feature_importance.png')
    plt.savefig(plot_path)
    plt.show()
    print(f"Feature importance plot saved to {plot_path}")
    
def generate_predictions(model, val_dataset, device='cpu', output_folder='./results/CNN', num_images=16):
    """
    Picks 16 random images from val_dataset, predicts their bounding boxes,
    draws them along with true bounding boxes, and saves the results in a grid format.
    """
    os.makedirs(output_folder, exist_ok=True)
    model.eval()

    # Define mean and std used for normalization
    mean = torch.tensor([0.5, 0.5, 0.5])
    std = torch.tensor([0.5, 0.5, 0.5])

    # Prepare a grid to store results
    grid_size = int(num_images ** 0.5)  # Assuming square grid (e.g., 4x4 for 16 images)
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(12, 12))

    with torch.no_grad():
        for idx in range(num_images):
            # Select a random sample
            random_idx = random.randint(0, len(val_dataset) - 1)
            image, extra_feats, true_bbox = val_dataset[random_idx]

            # Prepare inputs
            image_input = image.unsqueeze(0).to(device)  # Add batch dimension
            extra_feats_input = extra_feats.unsqueeze(0).to(device)

            # Get predictions
            pred_bbox = model(image_input, extra_feats_input)[0].cpu()  # Predicted bounding box in (cx, cy, w, h)
            pred_bbox = clamp_bbox_centerwh(pred_bbox.unsqueeze(0))[0]  # Clamp predictions to [0,1]
            pred_corner = center_to_corner(pred_bbox.unsqueeze(0))[0]  # Convert to (x_min, y_min, x_max, y_max)

            # Denormalize the image for visualization
            image_denorm = image * std[:, None, None] + mean[:, None, None]
            pil_image = transforms.ToPILImage()(image_denorm).convert('RGB')

            # Scale predicted bbox to pixel coordinates
            width, height = pil_image.size
            x_min, y_min, x_max, y_max = pred_corner
            x_min, x_max = x_min * width, x_max * width
            y_min, y_max = y_min * height, y_max * height

            # True bbox
            true_corner = center_to_corner(true_bbox.unsqueeze(0))[0]  # Convert to (x_min, y_min, x_max, y_max)
            x_min_t, y_min_t, x_max_t, y_max_t = true_corner
            x_min_t, x_max_t = x_min_t * width, x_max_t * width
            y_min_t, y_max_t = y_min_t * height, y_max_t * height

            # Draw true and predicted bounding boxes
            draw = ImageDraw.Draw(pil_image)
            draw.rectangle([x_min, y_min, x_max, y_max], outline='red', width=3)  # Predicted bbox
            draw.rectangle([x_min_t, y_min_t, x_max_t, y_max_t], outline='green', width=3)  # True bbox

            # Plot on the grid
            ax = axs[idx // grid_size, idx % grid_size]
            ax.imshow(pil_image)
            ax.axis('off')
            ax.set_title(f"Sample {random_idx}")

    # Save the grid of images
    grid_path = os.path.join(output_folder, "random_predictions_grid.png")
    plt.tight_layout()
    plt.savefig(grid_path)
    plt.show()
    print(f"Random predictions grid saved to {grid_path}")

def evaluate_iou(model, val_loader, device='cpu'):
    """
    Compute average IoU across the validation set for single bounding-box predictions.
    The target bounding boxes are in (cx, cy, w, h) [normalized].
    """
    model.eval()
    total_iou = 0.0
    count = 0

    with torch.no_grad():
        for (images, extra_feats, targets) in val_loader:
            images = images.to(device)
            extra_feats = extra_feats.to(device)
            targets = targets.to(device)

            preds = model(images, extra_feats)  # shape: (batch_size, 4)
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

def plot_training(batch_train_losses, batch_val_losses, batch_train_ious, batch_val_ious, output_folder='./results/CNN'):
    """
    Plots and saves continuous loss and IoU metrics for all batches across epochs.
    """
    os.makedirs(output_folder, exist_ok=True)
    train_batches = range(1, len(batch_train_losses) + 1)
    val_batches = range(1, len(batch_val_losses) + 1)

    # Plot Loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_batches, batch_train_losses, label='Train Loss', marker='o', linestyle='-', markersize=2)
    plt.plot(val_batches, batch_val_losses, label='Validation Loss', marker='x', linestyle='-', markersize=2)
    plt.title('Continuous Loss During Training and Validation')
    plt.xlabel('Batches (Cumulative)')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, 'continuous_loss_curve.png'))
    plt.show()

    # Plot IoU
    plt.figure(figsize=(10, 6))
    plt.plot(train_batches, batch_train_ious, label='Train IoU', marker='o', linestyle='-', markersize=2)
    plt.plot(val_batches, batch_val_ious, label='Validation IoU', marker='x', linestyle='-', markersize=2)
    plt.title('Continuous IoU During Training and Validation')
    plt.xlabel('Batches (Cumulative)')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, 'continuous_iou_curve.png'))
    plt.show()

# ---------------------------------------------------------------------
#  Main Function
# ---------------------------------------------------------------------
def main(lr=0.001, batch_size=8, num_epochs=1, early_stop_threshold=0.001, prompt_for_early_stop=True, device='cpu'):
    # File paths
    train_folder = './data/processed/train/'
    val_folder = './data/processed/test/'
    output_folder = './results/CNN'

    os.makedirs(output_folder, exist_ok=True)

    # Load processed datasets
    train_images = torch.load(os.path.join(train_folder, 'images.pt'), weights_only=True)
    train_features = torch.load(os.path.join(train_folder, 'features.pt'),weights_only=True)
    val_images = torch.load(os.path.join(val_folder, 'images.pt'))
    val_features = torch.load(os.path.join(val_folder, 'features.pt'))
    
    # Separate bounding boxes and additional features
    train_bboxes = train_features[:, :4]  # First 4 values are the bounding box
    train_extras = train_features[:, 4:]  # Remaining values are additional features
    val_bboxes = val_features[:, :4]
    val_extras = val_features[:, 4:]
    
    # Prepare Datasets and DataLoaders
    train_dataset = TensorDataset(train_images, train_extras, train_bboxes)
    val_dataset = TensorDataset(val_images, val_extras, val_bboxes)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model, optimizer, and loss
    model = CNN(extra_in=len(train_dataset[0][1]))  # Extra features dynamically determined
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)  # Add weight decay
    criterion = nn.SmoothL1Loss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    batch_train_losses, batch_train_ious = [], []
    batch_val_losses, batch_val_ious = [], []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Train
        train_losses, train_ious = train_model(model, train_loader, optimizer, criterion, num_epochs=1, device=device)
        batch_train_losses.extend(train_losses)
        batch_train_ious.extend(train_ious)

        # Validate
        val_losses, val_ious = validate_model(model, val_loader, criterion, device=device)
        batch_val_losses.extend(val_losses)
        batch_val_ious.extend(val_ious)

        # Update scheduler
        scheduler.step(val_losses[-1])  # Update learning rate based on the latest validation loss

    # Generate predictions
    generate_predictions(model, val_dataset, device=device, output_folder=output_folder)

    # Analyze feature importance
    feature_names = [
        "length", "wingspan", "area", 
        "wing_type_code", "wing_position_code", 
        "canard", "num_engines", "num_tailfins", "faa_class"]
    analyze_feature_importance(model, feature_names, output_folder)

    # Save model
    # avg_iou = evaluate_iou(model, val_loader, device=device)
    # print(f"Average Validation IoU: {avg_iou:.4f}")

    # Plot training progress
    plot_training(batch_train_losses, batch_val_losses, batch_train_ious, batch_val_ious, output_folder)
    print("Training completed!")

if __name__ == "__main__":
    main(lr=0.00001, batch_size=8, num_epochs=1, early_stop_threshold=0.001, prompt_for_early_stop=True, device='cpu')
