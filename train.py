from ultralytics import YOLO

# Load a pre-trained YOLOv8 model (you can use 'yolov8n.pt' for a smaller model)
model = YOLO('yolov8n.pt')  # Options: 'yolov8n.pt', 'yolov8s.pt', etc.

# Train the model on your dataset
model.train(
    data='./train_coco.yaml',
    epochs=1,
    lr0=0.0005,
    batch=16,
    optimizer='Adam',
    patience=5,
    augment=True,
    iou=0.5,
    conf=0.5
)

# Validate the model to check its performance on the validation set
results = model.val()

# Test the model on a new image
model.predict(source='.processed/test/images/130_10400100452B0100_tile_1436.png', save=True)