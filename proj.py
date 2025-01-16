from cmu_graphics import *
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
from hand_recognition2 import Net
import torch
from torchvision import transforms


class HandDetector:
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(static_image_mode=False,
                                          max_num_hands=2,
                                          min_detection_confidence=0.5,
                                          min_tracking_confidence=0.5)

    def detect(self, image):
        h, w = image.shape[:2]
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        landmarks = self.hands.process(imgRGB)
        if landmarks.multi_hand_landmarks is None:
            return np.array([])
        lms = np.array([[[lm.x * w, lm.y * h, lm.z] for lm in landmark.landmark] for landmark in landmarks.multi_hand_landmarks])
        return lms.astype(int)

class faceDetection:
    def __init__(self):
        self.face_detector = cv2.CascadeClassifier()
        self.face_detector.load("haarcascade_frontalface_alt2.xml")
        LBFmodel_name = ("lbf_face_landmarks.yaml")
        self.landmark_detector = cv2.face.createFacemarkLBF()
        self.landmark_detector.loadModel(LBFmodel_name)

    def detect(self, image):
        detection = self.face_detector.detectMultiScale(image)
        
        if detection is None or len(detection) == 0:
            return []
        _, landmarks = self.landmark_detector.fit(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), detection)
        return np.concatenate(landmarks, axis=0).astype(np.int64) 
          

class googlyEye:
    def __init__(self, x, y, width, height):
        self.x = int(x)
        self.y = int(y)
        self.width = int(width/3*2) + 1
        self.height = int(width/3*2) + 1


    def drawEyes(self):
        pil_image = Image.open("googly_eye.png")
        pil_image = pil_image.resize((self.width, self.height))
        image = CMUImage(pil_image)
        
        drawImage(image, self.x, self.y, align = "center")

    
class Hand:
    def __init__(self, left, right, top, bottom, shape):
        self.left = int(left)
        self.right = int(right)
        self.top = int(top)
        self.bottom = int(bottom)
        self.shape = 0
        biggest = -2
        
        for i in range(len(shape)):
            if shape[i] > biggest:
                self.shape = i
                biggest = shape[i]


        
    def drawEyes(self):
        left = self.left if self.left < self.right else self.right
        top = self.top if self.top < self.bottom else self.bottom
        drawRect(left, top, abs(self.right-self.left), abs(self.bottom-self.top), fill = None, border = "black")
        if self.shape == 1:
            drawLabel("paper", 300, 300, size = 20, fill = "black", bold = True)
        elif self.shape == 2:
            drawLabel("rock", 300, 300, size = 20, fill = "black", bold = True) 
        elif self.shape == 3:
            drawLabel("scissors", 300, 300, size = 20, fill = "black", bold = True)

def open_stream(app):
    app.video = cv2.VideoCapture(0)
    if not app.video.isOpened():
        app.quit()
    app.width = int(app.video.get(cv2.CAP_PROP_FRAME_WIDTH))
    app.height = int(app.video.get(cv2.CAP_PROP_FRAME_HEIGHT))

def update_image(app):
    sucess, image = app.video.read()
    if not sucess:
        app.quit()

    app.image = image[:, ::-1, ::-1]

def onAppStart(app):
    open_stream(app)
    update_image(app)
    app.detector = faceDetection()
    app.detectorHand = HandDetector()
    app.faces = app.detector.detect(app.image)
    app.hands = app.detectorHand.detect(app.image)
    app.gesture_recognizer = Net(4)
    app.gesture_recognizer.load_state_dict(torch.load("bn_hand_gesture_model_50.pt"))
    app.gesture_recognizer.eval()
    app.eyes = []
    app.handBox = []
    app.transformer = transforms.Compose([
                                    transforms.Grayscale(),
                                    transforms.Resize((128, 128)),
                                    transforms.ToTensor()
                                ])
        
 
def redrawAll(app):
    pil_image = Image.fromarray(app.image)
    image = CMUImage(pil_image)
    
    drawImage(image, 0, 0)
    
    for hand in app.hands:
        for i in range(len(hand)):
            point = hand[i]
            if i == 38:
                drawCircle(int(point[0]), int(point[1]), 5, fill = "black")
            else:
                drawCircle(int(point[0]), int(point[1]), 5, fill = "red")
    for face in app.faces:
        for i in range(len(face)):
            point = face[i]
            if i == 38:
                drawCircle(int(point[0]), int(point[1]), 5, fill = "black")
            else:
                drawCircle(int(point[0]), int(point[1]), 5, fill = "red")

    for handB in app.handBox:
        handB.drawEyes()
        
    for eye in app.eyes:
        eye.drawEyes()

    

def onStep(app):
    update_image(app)
    app.faces = app.detector.detect(app.image)
    app.eyes = []
    app.handBox = []
    app.hands = app.detectorHand.detect(app.image)
    
    for face in app.faces:
        leftCor, _ = face[36]
        rightCor, _ = face[49]
        _, topCor = face[37]
        _, bottomCor = face[41]

        width = abs(leftCor - rightCor)
        height = abs(topCor - bottomCor)
        xCor = (rightCor + leftCor) / 2
        yCor = (topCor + bottomCor) / 2
        
        app.eyes.append(googlyEye(xCor, yCor, width, height))

        leftCor, _ = face[42]
        rightCor, _ = face[45]
        _, topCor = face[43]
        _, bottomCor = face[47]

        width = abs(leftCor - rightCor)
        height = abs(topCor - bottomCor)
        xCor = (rightCor + leftCor) / 2
        yCor = (topCor + bottomCor) / 2


        
        app.eyes.append(googlyEye(xCor, yCor, width, height))
    
    
    for hand in app.hands:
        leftCors, _, _ = hand[4]
        rightCors, _, _= hand[20]
        _, topCors, _ = hand[12]
        _, bottomCors, _ = hand[0]

        for i in range(len(hand)):
            x, y, z = hand[i]
            if x < leftCors:
                leftCors = x
            if x > rightCors:
                rightCors = x
            if y < topCors:
                topCors = y
            if y > bottomCors:
                bottomCors = y
        
    
        pil_image = Image.fromarray(app.image)
        hand_patch = pil_image.crop((leftCors, topCors, rightCors, bottomCors))
        shape = app.gesture_recognizer(app.transformer(hand_patch).unsqueeze(0))[0]
        
        print(shape)
        app.handBox.append(Hand(leftCors, rightCors, topCors, bottomCors, shape))


def onKeyPress(app, key):
    if key == "q":
        app.quit()

def main():
    runApp()

main()