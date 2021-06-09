# 모델 필요

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from statistics import mode
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input
import matplotlib.pyplot as plt
from datetime import datetime
import os
import imutils

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

USE_WEBCAM = True # 웹캠 유무
emotion_model_path = 'trained_models/emotion_models/affectnet_mini_XCEPTION_128.h5'
emotion_labels = get_labels()
face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
emotion_classifier = load_model(emotion_model_path)
emotion_target_size = emotion_classifier.input_shape[1:3]

frame_window = 10
emotion_offsets = (20, 40)


emotion_window = []
video_capture = cv2.VideoCapture(0)

cap = None
if (USE_WEBCAM == True):
    cap = cv2.VideoCapture(0)
else: # 웹캠이 없을때 동영상으로
    cap = cv2.VideoCapture('./ifNoCam/a.mp4')  # https://www.youtube.com/watch?v=y2YUMPJATmg

while cap.isOpened():
    ret, bgr_image = cap.read()
    bgr_image = imutils.resize(bgr_image, width=1200) # 출력창 크기 조절
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40), flags=cv2.CASCADE_SCALE_IMAGE)

    for face_coordinates in faces:
        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        rgb_face = rgb_image[y1:y2, x1:x2, :]
        try:
            rgb_face = cv2.resize(rgb_face, (emotion_target_size))
        except:
            continue

        rgb_face = preprocess_input(rgb_face)
        rgb_face = np.expand_dims(rgb_face, 0)
        exp_prediction, etc_prediction = emotion_classifier.predict(rgb_face)
        emotion_probability = np.max(exp_prediction[0])
        emotion_label_arg = np.argmax(exp_prediction[0])
        emotion_text = emotion_labels[emotion_label_arg] + f';Val:{etc_prediction[0][0]:.5f};Aro:{etc_prediction[0][1]:.5f}'
        emotion_window.append(emotion_text)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue

        # ['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt'] [0,1,2,3,4,5,6]
        if emotion_text.split(';')[0] == emotion_labels[0]:
            color = emotion_probability * np.asarray((0, 255, 0))
        elif emotion_text.split(';')[0] == emotion_labels[1]:
            color = emotion_probability * np.asarray((255, 255, 0))
        elif emotion_text.split(';')[0] == emotion_labels[2]:
            color = emotion_probability * np.asarray((235, 107, 52))
        elif emotion_text.split(';')[0] == emotion_labels[3]:
            color = emotion_probability * np.asarray((38, 163, 151))
        elif emotion_text.split(';')[0] == emotion_labels[4]:
            color = emotion_probability * np.asarray((136, 38, 163))
        elif emotion_text.split(';')[0] == emotion_labels[5]:
            color = emotion_probability * np.asarray((163, 38, 138))
        elif emotion_text.split(';')[0] == emotion_labels[6]:
            color = emotion_probability * np.asarray((38, 42, 163))
        elif emotion_text.split(';')[0] == emotion_labels[7]:
            color = emotion_probability * np.asarray((140, 163, 38))
        else:
            color = emotion_probability * np.asarray((100, 100, 100))

        color = color.astype(int)
        color = color.tolist()

        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, emotion_mode, color, 0, -45, 0.5, 1)

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('2021-1 CapstonDesign_Emotion Recognition and Valence&Arousal_20154288', bgr_image)
    # cv2.imwrite(f'./output/{datetime.now().microsecond}.png',bgr_image)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
