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

class GestureDetector():
    def __init__(self, mode=False, maxHands=2, complexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.complexity=complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands=mp.solutions.hands
        self.hands=self.mpHands.Hands(self.mode, self.maxHands,
                                       self.complexity, self.detectionCon, self.trackCon)
        self.mpDraw=mp.solutions.drawing_utils
    def input_log(self, img, mode, number, draw=True):
        imgRGB=cv.cvtColor(img,cv.COLOR_BGR2RGB)
        self.results=self.hands.process(imgRGB)
    
        if self.results.multi_hand_landmarks:
            for handLms, handedness in zip(self.results.multi_hand_landmarks,
                                            self.results.multi_handedness):
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                    brect = calc_bounding_rect(img, handLms)
                    cv.rectangle(img, (brect[0], brect[1]),
                                  (brect[2], brect[3]), (255,0,0), 2)
                    hand_label = handedness.classification[0].label
                    if hand_label =="Left":
                        hand_label="Drone 1"
                    elif hand_label =="Right":
                        hand_label="Drone 2"
                    cv.putText(img, hand_label, (brect[0], brect[1] - 10),
                                cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    lmlist=calc_landmark_list(img, handLms)
                    pre_processed_lmlist = pre_process_landmark(lmlist)
                    logging_csv(number, mode, pre_processed_lmlist)
        return img
    def gesturepred(self, img, model, flagL, flagR, pred, prob1, prob2, draw=True):
        imgRGB=cv.cvtColor(img,cv.COLOR_BGR2RGB)
        self.results=self.hands.process(imgRGB)
    
        if self.results.multi_hand_landmarks:
            for handLms, handedness in zip(self.results.multi_hand_landmarks,
                                            self.results.multi_handedness):
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                    brect = calc_bounding_rect(img, handLms)
                    cv.rectangle(img, (brect[0], brect[1]),
                                  (brect[2], brect[3]), (255,0,0), 2)
                    hand_label = handedness.classification[0].label
                    #Too many hands spoil the drone
                    if hand_label =="Left":
                        hand_label="Drone 1"
                        cv.putText(img, hand_label, (brect[0], brect[1] - 10),
                                cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        if pred:
                            lmlist=calc_landmark_list(img, handLms)
                            pre_processed_lmlist = pre_process_landmark(lmlist)
                            prob1=model.predict(tf.expand_dims(pre_processed_lmlist, axis=0),verbose=0)
                        if prob1[0][0]!= 0:    
                            className = np.argmax(prob1)
                            labels={0:'start', 1:'up',2:'down',3:'left', 4:'right',5:'forward', 6:'backward',7:'land', 8:'front flip', 9:'land',10:'hover'}
                            if prob1[0][className] > 0.9:
                                if labels[className]=='start':
                                    flagL=True
                                if labels[className]=='land':
                                    flagL=False
                                if flagL:
                                    cName = labels[className] + ' ' + str(round(prob1[0][className]*100, 2)) + '%'
                                    cv.putText(img, "Drone 1: Capturing gestures", (10, 90),cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1,cv.LINE_AA)
                                    cv.putText(img, cName, (brect[0], brect[3] + 20), cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2, cv.LINE_AA)
                                    flagL=True
                                else:
                                    cv.putText(img, "Drone 1: Waiting to start", (10, 90),cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1,cv.LINE_AA)
                                    flagL=False
                                
                            else:
                                className = ''
                                cv.putText(img, className, (brect[0], brect[3] + 20), cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2, cv.LINE_AA)
                                if flagL:
                                    cv.putText(img, "Drone 1: Capturing gestures", (10, 90),cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1,cv.LINE_AA)
                                else:
                                    cv.putText(img, "Drone 1: Waiting to start", (10, 90),cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1,cv.LINE_AA)


                    elif hand_label =="Right":
                        hand_label="Drone 2"
                        cv.putText(img, hand_label, (brect[0], brect[1] - 10),
                                cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        if pred:
                            lmlist=calc_landmark_list(img, handLms)
                            pre_processed_lmlist = pre_process_landmark(lmlist)
                            prob2=model.predict(tf.expand_dims(pre_processed_lmlist, axis=0),verbose=0)
                        if prob2[0][0]!= 0:   
                            className = np.argmax(prob2)
                            labels={0:'start', 1:'up',2:'down',3:'left', 4:'right',5:'forward', 6:'backward',7:'land', 8:'front flip', 9:'land',10:'hover'}
                            if prob2[0][className] > 0.9:
                                if labels[className]=='start':
                                    flagR=True
                                if labels[className]=='land':
                                    flagR=False
                                if flagR:
                                    cName = labels[className] + ' ' + str(round(prob2[0][className]*100, 2)) + '%'
                                    cv.putText(img, "Drone 2: Capturing gestures", (365, 90),cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1,cv.LINE_AA)
                                    cv.putText(img, cName, (brect[0], brect[3] + 20), cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2, cv.LINE_AA)
                                    flagR= True
                                else:
                                    cv.putText(img, "Drone 2: Waiting to start", (365, 90),cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1,cv.LINE_AA)
                                    flagR= False 
                            else:
                                className = ''
                                cv.putText(img, className, (brect[0], brect[3] + 20), cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2, cv.LINE_AA)
                                if flagR:
                                    cv.putText(img, "Drone 2: Capturing gestures", (365, 90),cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1,cv.LINE_AA)
                                else:
                                    cv.putText(img, "Drone 2: Waiting to start", (365, 90),cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1,cv.LINE_AA)

        return img, flagL, flagR, prob1, prob2


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

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

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    max_value = max(list(map(abs, temp_landmark_list)))
    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

def logging_csv(number, mode, landmark_list):
    if mode == 0:
        pass
    elif mode == 1 and (0 <= number <= 10):
        csv_path = 'dataset.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    return


def main():
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

if __name__ == '__main__':
    main()


