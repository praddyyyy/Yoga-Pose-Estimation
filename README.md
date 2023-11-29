# Yoga-Pose

## Motivation:
The practice of yoga has gained significant popularity worldwide due to its ability to promote physical and mental well-being. 
Practicing yoga correctly is essential to avoid any potential harm or counterproductive effects. 
The development of AI-based technologies that can accurately recognize and guide yoga poses becomes crucial. 
By leveraging AI algorithms and computer vision techniques, this research aims to contribute to the advancement of pose detection in yoga. 
The ultimate goal is to provide practitioners with real-time feedback and recommendations to enhance their yoga practice, making it more beneficial and personalized.

## Introduction:
In today's fast-paced world, where stress levels are on the rise, yoga has garnered global interest as a means to achieve balance and well-being.
By combining AI technology and angle heuristics, this research proposes a rule-based technique called Pradhan. 
This technique allows users to upload images or perform yoga postures in front of a camera, enabling the classification of poses from a set of 10 pretrained postures. 
Through real-time video feed analysis, the system estimates yoga poses, providing practitioners with valuable assistance. 

## Objective:
The main objectives of the proposed work are to develop a rule-based technique for yoga pose estimation and provide real-time feedback to practitioners. 
The system's primary goal is to assist practitioners in practicing yoga correctly, ensuring that poses are performed safely and effectively.

## Pradhan Algorithm:
Input: Videos or images of yoga practitioners are provided to the system.
Feature Extraction: Frames are extracted from the input videos using OpenCV at regular intervals. Mediapipe Pose Estimation Python Framework is used to extract key points from these frames, and 8 joint angles are calculated based on these key points.
Pose Estimation: The calculated angles are fed into the rule-based system to estimate the pose among a set of 10 pre-trained poses.
3D Keypoint Calculation: The Pose Estimation module in Mediapipe employs a multi-stage pipeline to detect the presence of a human body and calculate the 3D coordinates of body keypoints, including shoulder, hip, elbow, and knee on both sides of the body.
Analysis of Keypoint Changes: By analyzing the changes in the position and visibility of keypoints over time, the system determines whether the yoga posture is being performed correctly.
Angle Calculation: To calculate angles between joints, such as A, B, and C, trigonometry is used based on their 2D coordinates. The arc tangent function is applied to determine the angle between the lines resulting from the points B-A and B-C. The calculated angle is adjusted to always represent the smallest angle between the line segments.
Output: The system provides the estimated yoga pose based on the rule-based analysis and angle calculations.

## Result:
A yoga pose estimation model achieved 93% accuracy on a dataset of 10 users performing 10 different poses (100 samples). 
The model accurately detected poses, considering user variances and pose execution variations. 
Although the study acknowledges limitations such as the small sample size and limited number of poses, the model demonstrates potential for accurate yoga pose estimation.
