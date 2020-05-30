import cv2
import sys
import os
from PIL import Image
import pandas as pd
import numpy as np
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "images")
MASK_DIR = os.path.join(IMG_DIR, "mask")
TEST_DIR = os.path.join(BASE_DIR, "Test")
TEST_IMG_DIR = os.path.join(TEST_DIR, "images_1")
TEST_VIDS_DIR = os.path.join(TEST_DIR, "vids")

cap = cv2.VideoCapture(os.path.join(TEST_VIDS_DIR,"test2.mp4"))
faceCascade = cv2.CascadeClassifier(os.path.join(BASE_DIR, "haarcascade_frontalface_alt.xml"))
recognizer = cv2.face.FisherFaceRecognizer_create()
recognizer.read(os.path.join(BASE_DIR,"model_trainner.yml"))

labels = {}
with open(os.path.join(BASE_DIR,"labels.pickle"),"rb") as file:
    og_labels = pickle.load(file)
    labels = {v:k for k,v in og_labels.items()}

while(cap.isOpened()):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.2, minNeighbors=3, minSize=(60,60))
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        size=(100,100)
        roi_gray = cv2.resize(roi_gray, size)
        id_, conf = recognizer.predict(roi_gray)
        font = cv2.FONT_HERSHEY_SIMPLEX
        name = labels[id_]
        color = (255,255,0)
        cv2.putText(frame,name,(x,y),font,1,color,2,cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("frame",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
