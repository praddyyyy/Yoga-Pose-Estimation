# Importing packages
import math
import PIL
import matplotlib
import numpy
import os
import datetime
import speech_recognition as sr
import operator
import random
import json
import tkinter
import subprocess
import ctypes
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


engine = pyttsx3.init()

initialization = "Your Yoga AI is Ready!"
engine.say(initialization)
engine.runAndWait()


def text2speech(text_file):
    pose = text_file
    f = open(pose, 'r')
    text = f.read()
    f.close()
    engine.setProperty('rate', 150)
    engine.say(text)
    engine.runAndWait()


# Function to calculate angles between two bones which will be used further.
def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360-angle

    return angle


cap = cv2.VideoCapture(0)

# Creating lable variable
label = "Unknown Pose"

# Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # STEP I
        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            # STEP II:Calculate angle between the joints
            left_elbow_angle = calculate_angle(
                left_shoulder, left_elbow, left_wrist)
            right_elbow_angle = calculate_angle(
                right_shoulder, right_elbow, right_wrist)
            left_shoulder_angle = calculate_angle(  
                left_hip, left_shoulder, left_elbow)
            right_shoulder_angle = calculate_angle(
                right_hip, right_shoulder, right_elbow)
            left_hip_angle = calculate_angle(
                left_knee, left_hip, left_shoulder)
            right_hip_angle = calculate_angle(
                right_knee, right_hip, right_shoulder)
            left_knee_angle = calculate_angle(left_ankle, left_knee, left_hip)
            right_knee_angle = calculate_angle(
                right_ankle, right_knee, right_hip)

            # STEP III: Classifing poses
            # Check if hands are staright or not
            if left_elbow_angle > 165 and left_elbow_angle < 195 and right_elbow_angle > 165 and right_elbow_angle < 195:
                if left_shoulder_angle > 80 and left_shoulder_angle < 110 and right_shoulder_angle > 80 and right_shoulder_angle < 110:
                    # Warrior II pose
                    if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:
                        if left_knee_angle > 90 and left_knee_angle < 120 or right_knee_angle > 90 and right_knee_angle < 120:
                            label = "warrior II Pose"
                            # text2speech("Warrior_II.txt")
                            engine.say(label)
                            engine.runAndWait()
                    # T Pose
                    if left_knee_angle > 160 and left_knee_angle < 195 and right_knee_angle > 160 and right_knee_angle < 195:
                        label = 'T Pose'
                        # text2speech("TPose.txt")
                        engine.say(label)
                        engine.runAndWait()
            # Tree Pose
            if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:
                if left_knee_angle > 315 and left_knee_angle < 335 or right_knee_angle > 25 and right_knee_angle < 45:
                    label = 'Tree Pose'
                    text2speech("treepose.txt")

            else:
                label = 'Unknown Pose'

        except:
            pass

        # Showing label on the output screen
        cv2.putText(image, label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 1, cv2.LINE_AA)

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(
                                      color=(0, 0, 255), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
