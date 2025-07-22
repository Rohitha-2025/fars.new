import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Pose detection model
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# List of available camera indexes (change these based on your system)
camera_indexes = [0, 1, 2]  # Modify if necessary
current_index = 0

# Setup Pose detection model
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.3, model_complexity=2)

# Open camera function
def open_camera(index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return None
    cap.set(3, 1280)  # Set width
    cap.set(4, 960)   # Set height
    return cap

# Check available cameras
def check_available_cameras():
    available_cameras = []
    for i in range(5):  # Check 5 possible camera indices
        cap = open_camera(i)
        if cap and cap.isOpened():
            available_cameras.append(i)
        if cap:
            cap.release()
    return available_cameras

# Check available cameras at the start
available_cameras = check_available_cameras()
print(f"Available cameras: {available_cameras}")

if not available_cameras:
    print("No cameras found. Please check your camera setup.")
    exit()

# Initialize the first camera
cap = open_camera(available_cameras[current_index])
cv2.namedWindow("Pose Detection", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "Camera not available", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        # Convert the image to RGB for Pose detection
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS
            )

    # Display camera feed with pose landmarks
    cv2.putText(frame, f"Camera Index: {available_cameras[current_index]}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Pose Detection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC key to quit
        break
    elif key == ord('t'):  # Press 'T' to toggle between cameras
        print("Toggling camera...")
        cap.release()  # Release the current camera
        current_index = (current_index + 1) % len(available_cameras)  # Toggle to next camera
        cap = open_camera(available_cameras[current_index])  # Reinitialize the new camera

cap.release()  # Ensure the camera is released at the end
cv2.destroyAllWindows()
