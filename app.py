# from pose_detection import detectPoseFromVideo, detectPose, pose_image
from pose_detection_angle import detectPose, pose_image
import cv2

if __name__ == "__main__":
    # image = cv2.imread('images/big-toe.jpg')
    image = cv2.imread('danush/hii.jpg')

    detectPose(image, pose_image, display=True)
    # detectPoseFromVideo()
