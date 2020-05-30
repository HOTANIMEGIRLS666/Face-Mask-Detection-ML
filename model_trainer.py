import cv2
import sys
import os
from PIL import Image
import pandas as pd
import numpy as np
import pickle

# DIRECTORY HANDLER
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "images")
TEST_DIR = os.path.join(BASE_DIR, "Test")
TEST_IMG_DIR = os.path.join(TEST_DIR, "images_1")
TEST_VIDS_DIR = os.path.join(TEST_DIR, "vids")
RESULT_NO_MASK = os.path.join(TEST_DIR, "result_no_mask")
RESULT_MASK = os.path.join(TEST_DIR, "result_mask")

# INIT
faceCascade = cv2.CascadeClassifier(os.path.join(BASE_DIR, "haarcascade_frontalface_alt.xml"))
#recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer = cv2.face.FisherFaceRecognizer_create()

current_id = 0
label_ids = {}
x_train = []
y_train = []
faces_test = {}

def detect_face(root, file, label, scale, minN, minS, faces_detected, current_id, DIR_WRITE):
    if file.endswith("png") or file.endswith("jpg"):
        path = os.path.join(root, file)
        if not label in label_ids:
            label_ids[label] = current_id
            current_id += 1
        id_ = label_ids[label]

        pil_image = Image.open(path).convert("L")
        image_array = np.array(pil_image, "uint8")
        faces = faceCascade.detectMultiScale(image_array, scaleFactor=scale, minNeighbors=minN, minSize=minS)
        for (x, y, w, h) in faces:
            detected_face_name = "face_" + str(faces_detected) + ".jpg"
            roi = image_array[y:y+h, x:x+w]
            size = (100,100)
            roi = cv2.resize(roi,size)
            x_train.append(roi)
            y_train.append(id_)
            faces_detected+=1
            faces_test[label] = faces_detected
            #cv2.imwrite(os.path.join(DIR_WRITE, detected_face_name), roi)
    return current_id, faces_detected

for root, dirs, files in os.walk(IMG_DIR):
    label = os.path.basename(root)
    faces_detected = 0
    for file in files:
        if label == "mask":
            current_id, faces_detected = detect_face(root, file, label, 1.025, 3, (60,60), faces_detected, current_id, RESULT_MASK)
        if label == "no_mask":
            current_id, faces_detected = detect_face(root, file, label, 1.08, 3, (100,100), faces_detected, current_id, RESULT_NO_MASK)

print(faces_test)

with open(os.path.join(BASE_DIR,"labels.pickle"), "wb") as file:
    pickle.dump(label_ids, file)

recognizer.train(x_train, np.array(y_train))
recognizer.save(os.path.join(BASE_DIR,"model_trainner.yml"))
