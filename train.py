from ultralytics import YOLO

# Load a pre-trained model (COCO)
model = YOLO('yolov8n.pt')  # Use 'yolov8s.pt', 'yolov8m.pt', etc., for different sizes

# Train on your dataset
model.train(data='coco.yaml', epochs=50, imgsz=640, batch=8)

results = model.val()
model.predict(source='./processed/test_images/2_1040010024CD5300_tile_124.png', save=True)
