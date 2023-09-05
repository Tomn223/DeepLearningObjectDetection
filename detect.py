import cv2
import torch
import pandas
import os

# vehicles_list = ["car", "truck", "bus", "motorcycle", "bicycle"]

# YOLO v5 class/category number ranges by supercategory
vehicles_range = [1,8]
person_num = 0
animals_range = [14, 23]

CONFIDENCE_THRESHOLD = 0.5

class Detector():

    def __init__(self):
        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.model.names

    def load_model(self):
        return torch.hub.load('ultralytics/yolov5', 'yolov5s')

    def predict(self, img):
        return self.model(img)
    
    def plot_boxes(self, results, img):
        for box in results.xyxy[0]:
            x1, y1, x2, y2, confidence, class_num = box
            x1, y1, x2, y2, class_num = int(x1), int(y1), int(x2), int(y2), int(class_num)

            if class_num == person_num:
                color = (0,255,0)
            elif class_num >= vehicles_range[0] and class_num <= vehicles_range[1]:
                color = (0,0,255)
            elif class_num >= animals_range[0] and class_num <= animals_range[1]:
                color = (255, 0, 0)
            else:
                # only detecting vehicles, pedestrians & animals
                continue

            class_name = self.CLASS_NAMES_DICT[class_num]

            if confidence > CONFIDENCE_THRESHOLD:
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)  # bounding box
                size_text, _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_PLAIN, 1, 2)
                w_text, h_text = size_text
                cv2.rectangle(img, (x1, y1-5), (x1+w_text, y1-5+h_text), color, -1)  # label background
                cv2.putText(img, class_name, (x1, y1-5+h_text), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2)  # label text

        return img
    

object_detector = Detector()

image_dir = os.path.join(os.getcwd(), "images") + "\\"
results_dir = os.path.join(os.getcwd(), "results") + "\\"
for image_file in os.listdir(image_dir):
    img = cv2.imread(image_dir + image_file)
    results = object_detector.predict(img)
    img = object_detector.plot_boxes(results, img)
    cv2.imwrite(results_dir + image_file, img)
    cv2.imshow("img", img)
    cv2.waitKey(0)