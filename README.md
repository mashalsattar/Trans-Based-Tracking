# Real-time Object Tracking with DETR

This project implements a real-time object tracking system using the pre-trained DETR (DEtection Transformer) model from Facebook Research. It utilizes a webcam feed and detects objects within each frame.

This project uses a pre-trained DETR model for object detection, not YOLOv8. YOLOv8 is a separate object detection model that is not used here.

# Project Details

This project focuses on demonstrating real-time object tracking through a simple application. It is not intended for production use and lacks functionalities like custom object class detection or model training.

# Labeling Images

Since this project utilizes a pre-trained DETR model, image labeling is not required. The model is already trained on a large dataset of various objects.

# System Requirements

Python 3.6 or later

A GPU is recommended for faster performance, but the code can run on CPU as well.

The aforementioned libraries need to be installed using pip installopencv-python torch torchvision Pillow numpy.
Running the Project

# Install the required libraries.

pip install OpenCV (cv2)

pip install PyTorch

pip install Torchvision

pip install Pillow (PIL Fork)

pip install NumPy

Run the script main.py. This will start the webcam feed and display objects detected in each frame with bounding boxes.

Press 'q' to exit the application.

# Disclaimer

The pre-trained DETR model used in this project might not detect all objects perfectly. Its performance depends on the training data and specific object types.
