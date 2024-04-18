import cv2
from imread_from_url import imread_from_url
import PIL.Image
import numpy as np
from yolov8 import YOLOv8

# Initialize yolov8 object detector
model_path = "./yolov8m.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.2, iou_thres=0.3)

def imread_from_path(img_path):
    image = PIL.Image.open(img_path)
    image = cv2.cvtColor(np.array(image, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    return image


# Read image
# img_url = "https://live.staticflickr.com/13/19041780_d6fd803de0_3k.jpg"
img_url = "./dog.jpg"
img = imread_from_path(img_url)
# img = imread_from_url(img_url)

# Detect Objects
boxes, scores, class_ids = yolov8_detector(img)

print(img)

# Draw detections
combined_img = yolov8_detector.draw_detections(img)
cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
cv2.imshow("Detected Objects", combined_img)
cv2.imwrite("doc/img/detected_objects.jpg", combined_img)
cv2.waitKey(0)
