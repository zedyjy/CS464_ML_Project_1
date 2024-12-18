from ultralytics import YOLO

# Load a pre-trained YOLOv8 model (you can use 'yolov8n.pt' for a smaller model)
model = YOLO('yolov8n.pt')  # Options: 'yolov8n.pt', 'yolov8s.pt', etc.

# Train the model on your dataset
model.train(data='./train_coco.yaml', epochs=10)


# Validate the model to check its performance on the validation set
results = model.val()

# Test the model on a new image
model.predict(source='./processed/test_images/105_104001003108D900_tile_47.png', save=True)