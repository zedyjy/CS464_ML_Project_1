import os
import requests
import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader
from detectron2.engine import inference_on_dataset

# Configuration Setup
config_dir = "configurations"
os.makedirs(config_dir, exist_ok=True)
config_file = os.path.join(config_dir, "model_config.yaml")

# Fetch Model Configuration
model_url = "https://github.com/facebookresearch/detectron2/raw/master/configs/_base_/models/Base-RCNN-FPN.yaml"
response = requests.get(model_url)
with open(config_file, "wb") as file:
    file.write(response.content)

# Model Initialization
cfg = get_cfg()
cfg.merge_from_file(config_file)
cfg.MODEL.WEIGHTS = "model_weights.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.DEVICE = "cuda"

predictor = DefaultPredictor(cfg)

# Data Setup
test_image = "test_image.png"
image = cv2.imread(test_image)

# Prediction and Visualization
output = predictor(image)
visualizer = Visualizer(image[:, :, ::-1], metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
result = visualizer.draw_instance_predictions(output["instances"].to("cpu"))

cv2.imshow("Inference Result", result.get_image()[:, :, ::-1])
cv2.waitKey(0)

# Dataset Registration
register_coco_instances("evaluation_dataset", {}, "path/to/annotations.json", "path/to/images")

# Evaluation
evaluator = COCOEvaluator("evaluation_dataset", cfg, False, output_dir="results/")
val_loader = build_detection_test_loader(cfg, "evaluation_dataset")
evaluation_metrics = inference_on_dataset(predictor.model, val_loader, evaluator)

print("Evaluation Metrics:", evaluation_metrics)
