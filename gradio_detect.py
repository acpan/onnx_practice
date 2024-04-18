import cv2
from imread_from_url import imread_from_url
import PIL.Image
import numpy as np
from yolov8 import YOLOv8
import gradio

# Initialize yolov8 object detector
model_path = "./yolov8m.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.2, iou_thres=0.3)


def detector(image):
    # Detect Objects
    boxes, scores, class_ids = yolov8_detector(image)
    # Draw detections
    combined_img = yolov8_detector.draw_detections(image)
    return combined_img

demo = gradio.Interface(detector, gradio.Image(), "image")
demo.launch()
