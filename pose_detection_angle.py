
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
from time import time
import math

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose_image = mp_pose.Pose(static_image_mode=True,
                          min_detection_confidence=0.3, model_complexity=2)
pose_video = mp_pose.Pose(static_image_mode=False,
                          min_detection_confidence=0.5, model_complexity=1)


mp_drawing = mp.solutions.drawing_utils

# Function to calculate angles between two bones which will be used further.


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
        np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360-angle

    return angle


def detectPose(image, pose, display=True):
    # Create a copy of the input image.
    output_image = image.copy()

    label = "Unknown Pose"

    # Convert the image from BGR into RGB format.
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform the Pose Detection.
    results = pose.process(imageRGB)

    # Retrieve the height and width of the input image.
    height, width, _ = image.shape

    # Initialize a list to store the detected landmarks.
    # landmarks = []

    # Check if any landmarks are detected.
    if results.pose_landmarks:

        # Draw Pose landmarks on the output image.
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)

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
        # FIXME check for hip angle for T Pose
        if left_elbow_angle > 165 and left_elbow_angle < 195 and right_elbow_angle > 165 and right_elbow_angle < 195:
            if left_shoulder_angle > 80 and left_shoulder_angle < 110 and right_shoulder_angle > 80 and right_shoulder_angle < 110:
                # Warrior II pose
                if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:
                    if left_knee_angle > 90 and left_knee_angle < 120 or right_knee_angle > 90 and right_knee_angle < 120:
                        label = "Warrior II Pose"
                        # text2speech("Warrior_II.txt")
                # T Pose
                if left_knee_angle > 160 and left_knee_angle < 195 and right_knee_angle > 160 and right_knee_angle < 195:
                    label = 'T Pose'

        if left_shoulder_angle > 210 and left_shoulder_angle < 240 and right_shoulder_angle > 210 and right_shoulder_angle < 240:
            if left_elbow_angle > 210 and left_elbow_angle < 240 and right_elbow_angle > 210 and right_elbow_angle < 240:
                if left_hip_angle > 165 and left_hip_angle < 195 and right_hip_angle > 165 and right_hip_angle < 195:
                    if left_knee_angle > 165 and left_knee_angle < 195 and right_knee_angle > 165 and right_knee_angle < 195:
                        label = 'Tree Pose'
                        angle = math.degree(math.atan2(right_elbow.y - right_shoulder.y, right_elbow.x - right_shoulder.x)-math.atan2(left_elbow.y - left_shoulder.y, left_elbow.x - left_shoulder.x))
                        angle = round(angle, 2)
        

        if left_shoulder_angle > 165 and left_shoulder_angle < 195 and right_shoulder_angle > 165 and right_shoulder_angle < 195:
            if left_elbow_angle > 165 and left_elbow_angle < 195 and right_elbow_angle > 165 and right_elbow_angle < 195:
                if left_hip_angle > 165 and left_hip_angle < 195 and right_hip_angle > 165 and right_hip_angle < 195:
                    if left_knee_angle > 165 and left_knee_angle < 195 and right_knee_angle > 165 and right_knee_angle < 195:
                        label = 'Upward Salute Pose'
        print("Left Shoulder:", left_shoulder_angle)
        print("Right Shoulder:", right_shoulder_angle)
        print("Left Elbow:", left_elbow_angle)
        print("Right Elbow:", right_elbow_angle)
        print("Left Hip:", left_hip_angle)
        print("Right Hip:", right_hip_angle)
        print("Left Knee:", left_knee_angle)
        print("Right Knee:", right_knee_angle)

        # if left_shoulder_angle > 120 and left_shoulder_angle < 150 and right_shoulder_angle > 120 and right_shoulder_angle < 150:
        #     if left_elbow_angle > 60 and left_elbow_angle < 120 and right_elbow_angle > 60 and right_elbow_angle < 120:
        #         if left_hip_angle > 0 and left_hip_angle < 30 and right_hip_angle > 0 and right_hip_angle < 30:
        #             if left_knee_angle > 165 and left_knee_angle < 195 and right_knee_angle > 165 and right_knee_angle < 195:
        #                 label = 'Big Toe Pose'
        #                 print(label)

        if left_shoulder_angle > 0 and left_shoulder_angle < 30 and right_shoulder_angle > 0 and right_shoulder_angle < 30:
            if left_elbow_angle > 165 and left_elbow_angle < 195 and right_elbow_angle > 165 and right_elbow_angle < 195:
                if left_hip_angle > 165 and left_hip_angle < 195 and right_hip_angle > 165 and right_hip_angle < 195:
                    if left_knee_angle > 165 and left_knee_angle < 195 and right_knee_angle > 165 and right_knee_angle < 195:
                        label = 'Mountain Pose'

        # right leg up
        if left_shoulder_angle > 80 and left_shoulder_angle < 140 and right_shoulder_angle > 65 and right_shoulder_angle < 120:
            if left_elbow_angle > 165 and left_elbow_angle < 195 and right_elbow_angle > 165 and right_elbow_angle < 180:
                if left_hip_angle > 40 and left_hip_angle < 90 and right_hip_angle > 150 and right_hip_angle < 180:
                    if left_knee_angle > 160 and left_knee_angle < 180 and right_knee_angle > 160 and right_knee_angle < 180:
                        label = 'Half Moon Pose'

        if left_shoulder_angle > 150 and left_shoulder_angle < 180 and right_shoulder_angle > 150 and right_shoulder_angle < 180:
            if left_elbow_angle > 70 and left_elbow_angle < 130 and right_elbow_angle > 70 and right_elbow_angle < 130:
                if left_hip_angle > 20 and left_hip_angle < 70 and right_hip_angle > 20 and right_hip_angle < 70:
                    if left_knee_angle > 160 and left_knee_angle < 180 and right_knee_angle > 160 and right_knee_angle < 180:
                        label = 'Dolphin Pose'

        if left_shoulder_angle > 150 and left_shoulder_angle < 180 and right_shoulder_angle > 150 and right_shoulder_angle < 180:
            if left_elbow_angle > 150 and left_elbow_angle < 180 and right_elbow_angle > 150 and right_elbow_angle < 180:
                if left_hip_angle > 110 and left_hip_angle < 150 and right_hip_angle > 110 and right_hip_angle < 160:
                    if (left_knee_angle > 90 and left_knee_angle < 130 and right_knee_angle > 160 and right_knee_angle < 180) or (left_knee_angle > 160 and left_knee_angle < 180 and right_knee_angle > 90 and right_knee_angle < 130):
                        label = 'High Lunge Pose'

        if left_shoulder_angle > 140 and left_shoulder_angle < 180 and right_shoulder_angle > 140 and right_shoulder_angle < 180:
            if left_elbow_angle > 150 and left_elbow_angle < 180 and right_elbow_angle > 150 and right_elbow_angle < 180:
                if (left_hip_angle > 80 and left_hip_angle < 130 and right_hip_angle > 150 and right_hip_angle < 180) or (left_hip_angle > 150 and left_hip_angle < 180 and right_hip_angle > 80 and right_hip_angle < 130):
                    if left_knee_angle > 150 and left_knee_angle < 180 and right_knee_angle > 150 and right_knee_angle < 180:
                        label = 'Warrior III Pose'

        if (left_shoulder_angle > 60 and left_shoulder_angle < 100 and right_shoulder_angle > 90 and right_shoulder_angle < 150) or (left_shoulder_angle > 90 and left_shoulder_angle < 150 and right_shoulder_angle > 60 and right_shoulder_angle < 100):
            if left_elbow_angle > 150 and left_elbow_angle < 180 and right_elbow_angle > 150 and right_elbow_angle < 180:
                if left_hip_angle > 150 and left_hip_angle < 180 and right_hip_angle > 150 and right_hip_angle < 180:
                    if left_knee_angle > 150 and left_knee_angle < 180 and right_knee_angle > 150 and right_knee_angle < 180:
                        label = 'Side Plank Pose'

        else:
            label = 'No Pose'

        # # Iterate over the detected landmarks.
        # for landmark in results.pose_landmarks.landmark:

        #     # Append the landmark into the list.
        #     landmarks.append((int(landmark.x * width), int(landmark.y * height),
        #                       (landmark.z * width)))

    # Check if the original input image and the resultant image are specified to be displayed.
    if display:

        # Display the original input image and the resultant image.
        plt.figure(figsize=[11, 11])
        # plt.subplot(122)
        plt.imshow(output_image[:, :, ::-1])

        plt.axis('off')

        # Also Plot the Pose landmarks in 3D.
        mp_drawing.plot_landmarks(
            results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
        
        mp_drawing.draw_landmarks(image,
                                  results.pose_landmarks,
                                  mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                  connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                  )      
        # image = cv2.putText(image, f"Angle: {angle}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA) 

    # Otherwise
    else:
        # Return the output image and the found landmarks.
        return output_image, label


def detectPoseFromVideo():

    # Initialize the VideoCapture object to read from the webcam.
    video = cv2.VideoCapture(0)

    # Initialize the VideoCapture object to read from a video stored in the disk.
    # video = cv2.VideoCapture('images/video.mp4')

    # Initialize a variable to store the time of the previous frame.
    time1 = 0

    # Iterate until the video is accessed successfully.
    while video.isOpened():

        # Read a frame.
        ok, frame = video.read()

        # Check if frame is not read properly.
        if not ok:
            # Break the loop.
            break

        # Flip the frame horizontally for natural (selfie-view) visualization.
        frame = cv2.flip(frame, 1)

        # Get the width and height of the frame
        frame_height, frame_width, _ = frame.shape

        # Resize the frame while keeping the aspect ratio.
        frame = cv2.resize(
            frame, (int(frame_width * (640 / frame_height)), 640))

        # Perform Pose landmark detection.
        frame, label = detectPose(frame, pose_video, display=False)

        # Set the time for this frame to the current time.
        time2 = time()

        # Check if the difference between the previous and this frame time &gt; 0 to avoid division by zero.
        if (time2 - time1) > 0:

            # Calculate the number of frames per second.
            frames_per_second = 1.0 / (time2 - time1)

            # Write the calculated number of frames per second on the frame.
            cv2.putText(frame, 'FPS: {}'.format(int(frames_per_second)),
                        (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

        # Update the previous frame time to this frame time.
        # As this frame will become previous frame in next iteration.
        time1 = time2
        cv2.putText(frame, label, (260, 30),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
        # Display the frame.
        cv2.imshow('Pose Detection', frame)

        # Wait until a key is pressed.
        # Retreive the ASCII code of the key pressed
        k = cv2.waitKey(1) & 0xFF

        # Check if 'ESC' is pressed.
        if (k == 27):

            # Break the loop.
            break

    # Release the VideoCapture object.
    video.release()

    # Close the windows.
    cv2.destroyAllWindows()
