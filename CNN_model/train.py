import os
import json
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw
from tqdm import tqdm

# CNN Model Definition
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc = nn.Linear(64 * 64 * 64, 4)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = x.view(-1, 64 * 64 * 64)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x


# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, image_folder, label_folder, transform=None):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.transform = transform

        self.image_files = sorted([f for f in os.listdir(self.image_folder) if f.endswith('.png')])
        self.label_files = sorted([f for f in os.listdir(self.label_folder) if f.endswith('.geojson')])

        print(f"Found {len(self.image_files)} images and {len(self.label_files)} labels.")

        if len(self.image_files) != len(self.label_files):
            raise ValueError("Mismatch between images and labels count!")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_files[idx])
        label_path = os.path.join(self.label_folder, self.label_files[idx])

        # Load image
        image = Image.open(img_path).convert('RGB')
        width, height = image.size

        # Load bounding box and normalize to [0, 1]
        bbox = load_bounding_box(label_path)
        x0, y0, x1, y1 = bbox
        x0, x1 = x0 / width, x1 / width
        y0, y1 = y0 / height, y1 / height
        normalized_bbox = [x0, y0, x1, y1]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(normalized_bbox, dtype=torch.float32)


# Utility Functions
def load_bounding_box(label_file):
    # Read the bounding box from the JSON file
    for f in os.listdir(label_file):
        if f.endswith('.geojson'):
            bbox = json.load(f)['features'][0]['geometry']['coordinates'][0]
    
    x_coords, y_coords = zip(*bbox)
    return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]



def train_model(model, train_loader, optimizer, criterion, num_epochs=1):
    model.train()
    train_losses = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}")
        for images, targets in progress_bar:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch + 1} Loss: {epoch_loss}")

    return train_losses


def validate_model(model, val_loader, criterion):
    model.eval()
    val_losses = []

    progress_bar = tqdm(val_loader, desc="Validating")
    with torch.no_grad():
        for images, targets in progress_bar:
            outputs = model(images)
            loss = criterion(outputs, targets)
            val_losses.append(loss.item())
            progress_bar.set_postfix(loss=loss.item())
    avg_loss = sum(val_losses) / len(val_losses)
    print(f"Validation Loss: {avg_loss}")
    return avg_loss


def generate_images_with_bounding_boxes(model, val_loader, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    model.eval()

    with torch.no_grad():
        for idx, (images, _) in enumerate(tqdm(val_loader, desc="Generating Bounding Boxes")):
            outputs = model(images)

            for i in range(len(outputs)):
                image = transforms.ToPILImage()(images[i]).convert('RGB')
                width, height = image.size

                # Scale bounding box back to pixel coordinates
                predicted_bbox = outputs[i].numpy()
                x0 = predicted_bbox[0] * width
                y0 = predicted_bbox[1] * height
                x1 = predicted_bbox[2] * width
                y1 = predicted_bbox[3] * height

                # Ensure valid coordinates
                x0, x1 = sorted([x0, x1])
                y0, y1 = sorted([y0, y1])

                # Draw the bounding box
                draw = ImageDraw.Draw(image)
                draw.rectangle([x0, y0, x1, y1], outline='red', width=2)

                # Save the image
                image_path = os.path.join(output_folder, f"image_{idx}_{i}.png")
                image.save(image_path)


def plot_training_process(train_losses, val_losses, num_epochs):
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Training Loss", marker='o')
    plt.plot(epochs, val_losses, label="Validation Loss", marker='o')
    plt.title("Training and Validation Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


# Main Script
if __name__ == "__main__":
    train_image_folder = '/Users/eylul/Documents/python/CS464_ML_Learning_Project/raw/train/PS-RGB_tiled'
    train_label_folder = '/Users/eylul/Documents/python/CS464_ML_Learning_Project/raw/train/geojson_aircraft_tiled'
    val_image_folder = '/Users/eylul/Documents/python/CS464_ML_Learning_Project/raw/test/PS-RGB_tiled'
    val_label_folder = '/Users/eylul/Documents/python/CS464_ML_Learning_Project/raw/test/geojson_aircraft_tiled'
    output_folder = '/Users/eylul/Documents/python/CS464_ML_Learning_Project/results'

    # Data preparation
    transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    train_dataset = CustomDataset(train_image_folder, train_label_folder, transform)
    val_dataset = CustomDataset(val_image_folder, val_label_folder, transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Model, loss, optimizer
    model = CNN()
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print(load_bounding_box(val_label_folder))
    # Training and validation
    print("Training started...")
    num_epochs = 5
    train_losses = train_model(model, train_loader, optimizer, criterion, num_epochs=num_epochs)
    val_loss = validate_model(model, val_loader, criterion)

    # Generate images with bounding boxes
    print("Generating bounding boxes...")
    generate_images_with_bounding_boxes(model, val_loader, output_folder)

    # Plot training process
    plot_training_process(train_losses, [val_loss] * num_epochs, num_epochs)

    print("Process complete!")