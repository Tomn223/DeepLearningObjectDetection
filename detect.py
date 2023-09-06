import cv2
import torch
import os

# YOLOv5 class/category number ranges by supercategory
VEHICLES_RANGE = [1,8]
PERSON_NUM = 0
ANIMALS_RANGE = [14, 23]

# how confident model must be to display object bounding box
CONFIDENCE_THRESHOLD = 0.5

class Detector():

    def __init__(self):
        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.model.names

    def load_model(self):
        return torch.hub.load('ultralytics/yolov5', 'yolov5s')

    def predict(self, img):
        return self.model(img)
    
    def plot_boxes(self, results, img, confidence_ovr=None):
        for box in results.xyxy[0]:
            x1, y1, x2, y2, confidence, class_num = box
            x1, y1, x2, y2, class_num = int(x1), int(y1), int(x2), int(y2), int(class_num)

            if class_num == PERSON_NUM:
                color = (0,255,0)
            elif class_num >= VEHICLES_RANGE[0] and class_num <= VEHICLES_RANGE[1]:
                color = (0,0,255)
            elif class_num >= ANIMALS_RANGE[0] and class_num <= ANIMALS_RANGE[1]:
                color = (255, 0, 0)
            else:
                # only detecting vehicles, pedestrians & animals
                continue

            class_name = self.CLASS_NAMES_DICT[class_num]

            # needed for confidence tests
            conf_threshold = confidence_ovr if confidence_ovr is not None else CONFIDENCE_THRESHOLD

            if confidence > conf_threshold:
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)  # bounding box
                size_text, _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_PLAIN, 1, 2)
                w_text, h_text = size_text
                cv2.rectangle(img, (x1, y1-5), (x1+w_text, y1-5+h_text), color, -1)  # label background
                cv2.putText(img, class_name, (x1, y1-5+h_text), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2)  # label text

        return img

def all_image_object_detection(object_detector, image_dir, save_dir):

    for image_file in os.listdir(image_dir):
        img = cv2.imread(image_dir + image_file)
        results = object_detector.predict(img)
        img = object_detector.plot_boxes(results, img)
        cv2.imwrite(save_dir + image_file, img)
        cv2.imshow("img", img)
        cv2.waitKey(0)

def confidence_tests(object_detector, image_file, save_dir, conf_test_list):

    for conf_threshold in conf_test_list:
        img = cv2.imread(image_dir + image_file) # to prevent overwriting same image
        results = object_detector.predict(img)
        img = object_detector.plot_boxes(results, img, confidence_ovr=conf_threshold)
        cv2.imwrite(save_dir + str(conf_threshold) + "-conf" + image_file, img)
        cv2.imshow("img", img)
        cv2.waitKey(0)

object_detector = Detector()

image_dir = os.path.join(os.getcwd(), "images") + "\\"
results_dir = os.path.join(os.getcwd(), "results") + "\\"
confidence_test_dir = os.path.join(os.getcwd(), "conf_test_results") + "\\"

test_image_dir = os.path.join(os.getcwd(), "test_images") + "\\"
test_results_dir = os.path.join(os.getcwd(), "test_results") + "\\"

all_image_object_detection(object_detector, image_dir, results_dir) 

# confidence_test_list = [0.1, 0.25, 0.5, 0.75, 0.9]
# confidence_tests(object_detector, "1.jpg", confidence_test_dir, confidence_test_list)

all_image_object_detection(object_detector, test_image_dir, test_results_dir) 


