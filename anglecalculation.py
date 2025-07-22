import cv2
import math
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from time import time

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

# Set up the Hand detector (MediaPipe Hands model)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3, min_tracking_confidence=0.3)

# Set up the Pose detector (MediaPipe Pose model)
pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.3, model_complexity=2)

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    # Convert the coordinates from 3D to 2D
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    # Calculate the angle between the vectors
    ab = a - b
    bc = c - b
    cos_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    angle = np.arccos(cos_angle)
    angle = np.degrees(angle)
    return angle

# Function to detect hands
def detHands(image, hands_model, display=True):
    op_img = image.copy()
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    res = hands_model.process(img_rgb)

    img_h, img_w, _ = image.shape
    landmarks = []

    if res.multi_hand_landmarks:
        for hand_landmarks in res.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image=op_img,
                landmark_list=hand_landmarks,
                connections=mp_hands.HAND_CONNECTIONS
            )

            for landmark in hand_landmarks.landmark:
                landmarks.append((int(landmark.x * img_w), int(landmark.y * img_h), int(landmark.z * img_w)))

    if display:
        fig = plt.figure(figsize=[8, 8])
        plt.subplot(1, 2, 1)
        plt.imshow(image[:, :, ::-1])
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(op_img[:, :, ::-1])
        plt.title("Hand Detected")
        plt.axis('off')
    else:
        return op_img, landmarks

# Function to detect body pose
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

    return op_img, landmarks

# Video capture setup
current_camera = 0  # Default to Mac webcam
video = cv2.VideoCapture(current_camera)  # Start with the default camera
cv2.namedWindow('Pose and Hand detection', cv2.WINDOW_NORMAL)
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

    # Detect hand landmarks
    frame, _ = detHands(frame, hands, display=False)

    # Detect pose landmarks (body)
    frame, landmarks = detPose(frame, pose_video, display=False)

    if landmarks:
        # Right leg (hip, knee, ankle)
        hip = landmarks[24]  # Right hip
        knee = landmarks[26]  # Right knee
        ankle = landmarks[28]  # Right ankle

        # Calculate knee angle
        angle = calculate_angle(hip, knee, ankle)

        # Show the knee angle on the screen
        cv2.putText(frame, f'Knee Angle: {int(angle)}', (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Calculate FPS
    time2 = time()
    if (time2 - time1) > 0:
        fps = 1.0 / (time2 - time1)
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

    time1 = time2

    # Display frame
    cv2.imshow('Pose and Hand detection', frame)

    # Handle keypress to toggle cameras
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key to exit
        break
    elif key == ord('t'):  # 't' to toggle between cameras
        current_camera = 1 if current_camera == 0 else 0  # Toggle camera index
        video.release()  # Release current camera
        video = cv2.VideoCapture(current_camera)  # Switch to the other camera

video.release()
cv2.destroyAllWindows()
