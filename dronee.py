import cv2 as cv
import mediapipe as mp
import time
import numpy as np
import itertools
import csv
import copy
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
from djitellopy import Tello

from main import GestureDetector

def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 to 9
        number = key - 48
    elif key==97:   # a for 10
        number=10
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 109:  # m
        mode = 2
    if key == 118:  # v
        mode = 3
    if key == 101:  # e
        mode = 0
    return number, mode

def draw_info(image, mode, number):
    mode_string = 'Logging Key Point'
    if mode==1:
        cv.putText(image, "MODE:" + mode_string, (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image

cap=cv.VideoCapture(0)
pTime=0
Ctime=0
detector=GestureDetector()
mode=0
model = tf.keras.models.load_model('gestures_model.keras')  
ctr = 1
flagL = False
flagR = False
prob1=np.zeros((1,1))
prob2=np.zeros((1,1))
while True:
    # tello = Tello()
    # tello.connect()
    ret, img=cap.read()
    img = cv.flip(img, 1)
    key=cv.waitKey(1)
    if key==27: #ESC
        break
    number, mode = select_mode(key, mode)
    if mode==0:
        cv.putText(img, "Select Mode", (270, 20),
                cv.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 0), 1,
                cv.LINE_AA)
    elif mode==2:
        cv.putText(img, "Hand Gesture Mode", (270, 20),
                cv.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 0), 1,
                cv.LINE_AA)
        pred=False
        if ctr%5 == 0:
            ctr = 0
            pred=True
        ctr+=1

        # img=detector.input_log(img,mode,number)

        img, flagL, flagR, prob1, prob2 = detector.gesturepred(img,model,flagL,flagR,pred,prob1,prob2)

    elif mode==3:
        cv.putText(img, "Voice Mode", (270, 20),
                cv.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 0), 1,
                cv.LINE_AA)
    
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv.putText(img, str(int(fps)),(10,70),
                cv.FONT_HERSHEY_COMPLEX,2,(255,0,255),3)

    img=draw_info(img, mode, number)

    cv.imshow('Image',img)