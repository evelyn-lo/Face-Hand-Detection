# Face Hand Detection, Googly Eyes Filters, and Hand Gesture Identification

This project implements real-time googly eyes filter and hand gesture identification using Python, OpenCV, MediaPipe, and deep learning techniques. Only the edited file called **project.py** related to the **googly eyes** filter and **hand gesture training** is included.

## **Project Overview**:

- **Googly Eyes Filter**:
  - The project uses **`ibf_face_landmarks`** to detect facial features from live video feed.
  - Through object-oriented programming (OOP), the class **`googlyEye`** adds a googly eyes filter on the detected face, positioning it based on facial landmark detection.
  - **`self.face_detector`** and **`self.landmark_detector`** in the **`faceDetection`** class handle face detection and landmarking to guide the placement of the filter.

- **Hand Gesture Recognition**:
  - Hand gestures (rock, paper, and scissors) are identified using a convolutional neural network (CNN) trained with uploaded images.
  - Hand gestures are boxed and labeled using the **`Hand`** class, which determines whether the gesture is rock, paper, or scissors.
  - Detection and landmarking for hands are carried out using **`mp.solutions.hands`** from MediaPipe within the **`HandDetector`** class.

## **Key Features**:
- Real-time hand gesture and face detection with webcam.
- Integration of deep learning models for accurate gesture recognition in live video feed.
- Utilizes computer vision libraries such as OpenCV and MediaPipe for hand and facial landmark extraction.
- Object-Oriented Programming (OOP) to modularize face detection, hand detection, and filter rendering.
