import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
from torchvision.ops import box_iou
from PIL import Image, ImageDraw
from datetime import datetime
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
    def __init__(self):
        """
        CNN model for bounding box regression using only image features.
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

        # After the third pool, if input is 512x512 => output size is 64x64 with 64 channels => 64 * 64 * 64
        self.flat_dim = 32 * 32 * 64

        # Final fully connected layer for bounding box regression
        self.fc_final = nn.Linear(self.flat_dim, 4)

    def forward(self, x):
        """
        :param x: image tensor (B, 3, 512, 512)
        :return: bounding box (B, 4) => (cx, cy, w, h)
        """
        # CNN for image
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = x.view(-1, self.flat_dim)

        # Bounding box regression
        out = self.fc_final(x)
        return out

# ---------------------------------------------------------------------
#  Model Training and Evaluation
# ---------------------------------------------------------------------
def train_model(model, train_loader, optimizer, criterion, device='cpu'):
    model.to(device)
    model.train()

    total_loss = 0.0
    total_iou = 0.0
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

        total_loss += loss.item()
        total_iou += iou
        num_batches += 1

        progress_bar.set_postfix(loss=loss.item(), iou=iou)

    avg_loss = total_loss / num_batches
    avg_iou = total_iou / num_batches
    return avg_loss, avg_iou

def validate_model(model, val_loader, criterion, device='cpu'):
    model.to(device)
    model.eval()

    total_loss = 0.0
    total_iou = 0.0
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

            total_loss += loss.item()
            total_iou += iou
            num_batches += 1

            progress_bar.set_postfix(loss=loss.item(), iou=iou)

    avg_loss = total_loss / num_batches
    avg_iou = total_iou / num_batches
    return avg_loss, avg_iou

# ---------------------------------------------------------------------
#  Resluts and Analysis
# ---------------------------------------------------------------------
    
def generate_predictions(model, val_dataset, device='cpu', output_folder='./results/CNN', num_images=16):
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
            image, true_bbox = val_dataset[random_idx]

            image_input = image.unsqueeze(0).to(device)

            pred_bbox = model(image_input)[0].cpu()
            pred_bbox = clamp_bbox_centerwh(pred_bbox.unsqueeze(0))[0]
            pred_corner = center_to_corner(pred_bbox.unsqueeze(0))[0]

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

def plot_training(train_losses, val_losses, train_ious, val_ious, output_folder='./results/CNN'):
    """
    Plots and saves epoch-level loss and IoU metrics for training and validation.
    """
    os.makedirs(output_folder, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    # Plot Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss', marker='o', linestyle='-', markersize=4)
    plt.plot(epochs, val_losses, label='Validation Loss', marker='x', linestyle='-', markersize=4)
    plt.title('Loss During Training and Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, 'epoch_loss_curve.png'))
    plt.show()

    # Plot IoU
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_ious, label='Train IoU', marker='o', linestyle='-', markersize=4)
    plt.plot(epochs, val_ious, label='Validation IoU', marker='x', linestyle='-', markersize=4)
    plt.title('IoU During Training and Validation')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, 'epoch_iou_curve.png'))
    plt.show()

# ---------------------------------------------------------------------
#  Main Function
# ---------------------------------------------------------------------
def main(lr=0.001, batch_size=16, num_epochs=1, device='cpu'):
    # File paths
    train_folder = './data/processed/train/'
    val_folder = './data/processed/test/'
    
    # Create a unique folder for each run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Current date and time
    run_name = f"lr_({lr})_bs_({batch_size})_epochs_({num_epochs})_({timestamp})"  # Unique run name
    output_folder = os.path.join(r"./results/CNN", run_name)  # Combine base folder with run name
    os.makedirs(output_folder, exist_ok=True) 

    # Load processed datasets
    train_images = torch.load(os.path.join(train_folder, 'images.pt'), weights_only=True)
    train_features = torch.load(os.path.join(train_folder, 'features.pt'),weights_only=True)
    val_images = torch.load(os.path.join(val_folder, 'images.pt'), weights_only=True)
    val_features = torch.load(os.path.join(val_folder, 'features.pt'), weights_only=True)
    
    # Separate bounding boxes and additional features
    train_bboxes = train_features[:, :4]  # First 4 values are the bounding box
    val_bboxes = val_features[:, :4]
    
    # Prepare Datasets and DataLoaders
    train_dataset = TensorDataset(train_images, train_bboxes)
    val_dataset = TensorDataset(val_images, val_bboxes)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model, optimizer, and loss
    model = CNN()  # Extra features dynamically determined
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)  # Add weight decay
    criterion = nn.SmoothL1Loss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    epoch_train_losses = []
    epoch_train_ious = []
    epoch_val_losses = []
    epoch_val_ious = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Train
        train_loss, train_iou = train_model(model, train_loader, optimizer, criterion, device=device)
        epoch_train_losses.append(train_loss)
        epoch_train_ious.append(train_iou)

        # Validate
        val_loss, val_iou = validate_model(model, val_loader, criterion, device=device)
        epoch_val_losses.append(val_loss)
        epoch_val_ious.append(val_iou)

        # Update scheduler
        scheduler.step(val_loss)  # Update learning rate based on the latest validation loss

        # Print epoch results
        print(f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")

    # Plot training progress
    plot_training(epoch_train_losses, epoch_val_losses, epoch_train_ious, epoch_val_ious, output_folder)

    # Generate predictions
    generate_predictions(model, val_dataset, device=device, output_folder=output_folder)

    # Save model
    model_path = os.path.join(output_folder, 'model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Evaluate final IoU
    # avg_iou = evaluate_iou(model, val_loader, device=device)
    # print(f"Average Validation IoU: {avg_iou:.4f}")
    
    print("Training completed!")

if __name__ == "__main__":
    main(lr=0.001, batch_size=4, num_epochs=50, device='cpu')
