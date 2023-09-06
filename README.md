# DeepLearningObjectDetection
Object Detection from images in python using YOLOv5

## Usage
Packages required: OpenCV2, PyTorch


1. Clone the repository
2. Add images to be processed in images folder (.jpg extension)
3. Run the following command in your terminal:
 ```
  python detect.py
 ```
4. Press any key to move on to next image, or view processed images in results folder

## Rationale
A pretrained deep learning model seemed like the obvious choice here, because it would take loads of data and training time to make our own model using PyTorch.


After doing some research on various image processing and object detection models, I landed on YOLOv5. It had an object detection model with categories for every object I needed to detect (cars, people, animals)  and also seemed really easy to configure, making it perfect for this challenge.


Using PyTorch Hub allows us to download the pretrained YOLOv5 model (the model architecture and pretrained weights) and cache it for future use. We can then run the model on an image, which gives us a list of information regarding detected objects, like the coordinates of the bounding box, the YOLOv5 class number & name, and the confidence level.


I opted to use the YOLOv5s model, for which you can see some information about the model architecture/specs below:
```
YOLOv5s summary: 213 layers, 7225885 parameters, 0 gradients
```

Depending on your hardware/needs the other sizes of the model may be a better fit for you. Information on the other YOLOv5 models can be found here:
https://pytorch.org/hub/ultralytics_yolov5/


I decided to group the objects into vehicles, pedestrians, and animals, and color their bounding boxes red, green, and blue, respectively.
## Results
| Image | Result |
| --- | --- |
| ![Image Not Found](/images/1.jpg) | ![Image Not Found](/results/1.jpg) |
| ![Image Not Found](/images/2.jpg) | ![Image Not Found](/results/2.jpg) |
| ![Image Not Found](/images/4.jpg) | ![Image Not Found](/results/4.jpg) |

## Challenges/Improvements
One of the main challenges in this project was deciding which deep learning model to use (other options include Fast R-CNN and SSD). I decided on YOLO because I found lots of resources on it and it seemed relatively easy to implement. However, other models/architectures could be better for this application, but I would need to run more extensive tests (with some form of benchmarking) to see if this is the case.


Another challenge was setting the confidence threshold. This determines how confident we need the model to be that it has correctly detected and classified an object for us to display its bounding box. I initially set it very high (0.8) but found that this led to some obvious (to a human) objects to be missed. I ended up trying a range of values, the results of which can be seen below, and eventually settled on 0.5. This value represents a tradeoff, as we can detect many more things in the image if we use a low threshold (some being correct) at the expense of including some false classifications. (A funny example is the Starbucks logo being labeled a person. Is it really wrong though?).

| Confidence Threshold | Result |
| --- | --- |
| 0.25 | ![Image Not Found](/conf_test_results/0.25-conf1.jpg) |
| 0.50 | ![Image Not Found](/conf_test_results/0.5-conf1.jpg) |
| 0.75 | ![Image Not Found](/conf_test_results/0.75-conf1.jpg) |

## Links
- https://pytorch.org/hub/ultralytics_yolov5/
- https://pytorch.org/docs/stable/hub.html
- https://wandb.ai/onlineinference/YOLO/reports/YOLOv5-Object-Detection-on-Windows-Step-By-Step-Tutorial---VmlldzoxMDQwNzk4
- https://towardsdatascience.com/understanding-ssd-multibox-real-time-object-detection-in-deep-learning-495ef744fab
- https://towardsdatascience.com/understanding-and-implementing-faster-r-cnn-a-step-by-step-guide-11acfff216b0
- https://iq.opengenus.org/yolov5/
