import cv2
import math
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from time import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose  # Moved to top

def detPose(image, pose, display=True):
    op_img = image.copy()
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    res = pose.process(img_rgb)

    img_h, img_w, _ = image.shape
    landmarks = []

    if res.pose_landmarks:
        mp_drawing.draw_landmarks(
            image=op_img,
            landmark_list=res.pose_landmarks,
            connections=mp_pose.POSE_CONNECTIONS
        )

        for landmark in res.pose_landmarks.landmark:
            landmarks.append((int(landmark.x * img_w), int(landmark.y * img_h), int(landmark.z * img_w)))

    if display:
        fig = plt.figure(figsize=[8, 8])
        plt.subplot(1, 2, 1)
        plt.imshow(image[:, :, ::-1])
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(op_img[:, :, ::-1])
        plt.title("Pose Detected")
        plt.axis('off')

        if res.pose_world_landmarks:
            mp_drawing.plot_landmarks(res.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
    else:
        return op_img, landmarks

pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.3, model_complexity=2)

video = cv2.VideoCapture(0)  
cv2.namedWindow('Pose detection', cv2.WINDOW_NORMAL)
video.set(3, 1280)
video.set(4, 960)

time1 = 0

while video.isOpened():
    ok, frame = video.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape
    new_w = int(frame_w * 640 / frame_h)
    frame = cv2.resize(frame, (new_w, 640))

    frame, _ = detPose(frame, pose_video, display=False)

    time2 = time()
    if (time2 - time1) > 0:
        fps = 1.0 / (time2 - time1)
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

    time1 = time2

    cv2.imshow('Pose detection', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

video.release()
cv2.destroyAllWindows()