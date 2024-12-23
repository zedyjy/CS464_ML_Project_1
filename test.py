from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO('./yolov8n.pt')  # Provide the path to the trained model file

# Test the model on a new image
results = model.predict(source='/Users/cagin_aydin/ML_Project/CS464_ML_Project_1/processed/test/images/130_10400100452B0100_tile_1436.png', save=True)

# Results is a list, get the first result (in this case, it's a single image, so we access the first element)
result = results[0]  # This is the result for the single image

# Show the predictions on the image
result.show()  # Display the image with the detections
