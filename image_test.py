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

faceCascade = cv2.CascadeClassifier(os.path.join(BASE_DIR, "haarcascade_frontalface_alt.xml"))
recognizer = cv2.face.FisherFaceRecognizer_create()
recognizer.read(os.path.join(BASE_DIR,"model_trainner.yml"))
print(recognizer.getMean())
labels = {}
with open(os.path.join(BASE_DIR,"labels.pickle"),"rb") as file:
    og_labels = pickle.load(file)
    labels = {v:k for k,v in og_labels.items()}

for (_x) in range(50):
    imgName = str(_x) + "-with-mask.jpg"
    img = cv2.imread(os.path.join(TEST_IMG_DIR, imgName))
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.025, minNeighbors=3, minSize=(60,60))
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        size=(100,100)
        roi_gray = cv2.resize(roi_gray, size)
        id_, conf = recognizer.predict(roi_gray)
        name = labels[id_]
        cv2.putText(img, name, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2, cv2.LINE_AA)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow("image", img)

    if cv2.waitKey(200) & 0xFF == ord('q'):
        break
    cv2.waitKey(0)

cv2.destroyAllWindows()
