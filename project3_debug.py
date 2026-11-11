import cv2
import math
import mediapipe as mp
import numpy as np
from time import time
import joblib
import sys

print("✓ Step 1: Imports successful", flush=True)

try:
    print("• Loading model...", flush=True)
    model = joblib.load('exercise_classifier_rf.joblib')
    print("✓ Step 2: Model loaded successfully", flush=True)
except Exception as e:
    print(f"✗ Error loading model: {e}", flush=True)
    model = None

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

print("✓ Step 3: MediaPipe initialized", flush=True)

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.3, min_tracking_confidence=0.3)
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.3, model_complexity=2)

print("✓ Step 4: Hands and Pose detectors created", flush=True)

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ab = a - b
    bc = c - b
    cosine_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def extract_landmarks(landmarks):
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()

def preprocess_landmarks(landmarks):
    coords = landmarks.reshape(-1, 3)
    min_vals = coords.min(axis=0)
    max_vals = coords.max(axis=0)
    norm = (coords - min_vals) / (max_vals - min_vals + 1e-6)
    return norm.flatten()

def get_feedback_and_count(exercise, angles, rep_count, stage):
    feedback = ""

    if exercise == "squat":
        knee_angle = angles.get("knee", 180)
        if stage == "up" and knee_angle < 90:
            stage = "down"
            rep_count += 1
            feedback = "Good squat rep!"
        elif stage == "down" and knee_angle > 160:
            stage = "up"
            feedback = "Stand up straight!"

    elif exercise == "pushup":
        elbow_angle = angles.get("elbow", 180)
        if stage == "up" and elbow_angle < 90:
            stage = "down"
            rep_count += 1
            feedback = "Good pushup rep!"
        elif stage == "down" and elbow_angle > 160:
            stage = "up"
            feedback = "Keep your back straight!"

    elif exercise == "dumbbell_press":
        elbow_angle = angles.get("elbow", 180)
        if stage == "up" and elbow_angle < 70:
            stage = "down"
            rep_count += 1
            feedback = "Good dumbbell press rep!"
        elif stage == "down" and elbow_angle > 150:
            stage = "up"
            feedback = "Lower slowly!"

    elif exercise == "lunge":
        knee_angle = angles.get("knee", 180)
        if stage == "up" and knee_angle < 90:
            stage = "down"
            rep_count += 1
            feedback = "Good lunge rep!"
        elif stage == "down" and knee_angle > 160:
            stage = "up"
            feedback = "Step back up!"

    elif exercise == "plank":
        feedback = "Hold steady!"

    else:
        feedback = "Exercise not recognized"

    return rep_count, stage, feedback


def draw_info_panel(frame, lines, x=10, y=10, padding=8, bg_color=(20, 20, 20), alpha=0.6):
    """Draw a semi-opaque info panel with multiple lines of text at (x,y)."""
    overlay = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    w = frame.shape[1]
    font_scale = max(0.6, min(1.0, w / 1000.0))
    thickness = 2

    max_w = 0
    line_height = 0
    sizes = []
    for line in lines:
        (tw, th), _ = cv2.getTextSize(str(line), font, font_scale, thickness)
        sizes.append((tw, th))
        if tw > max_w:
            max_w = tw
        if th > line_height:
            line_height = th

    box_w = max_w + padding * 2
    box_h = (line_height + 8) * len(lines) + padding * 2

    box_w = min(box_w, w - x - 10)
    box_h = min(box_h, frame.shape[0] - y - 10)

    cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), bg_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    yy = y + padding + line_height
    for i, line in enumerate(lines):
        cv2.putText(frame, str(line), (x + padding, yy - 4), font, font_scale, (255, 255, 255), thickness)
        yy += line_height + 8

print("✓ Step 5: Opening webcam...", flush=True)

current_camera = 0
cap = cv2.VideoCapture(current_camera)

if not cap.isOpened():
    print("✗ ERROR: Cannot open webcam!", flush=True)
    sys.exit(1)

print("✓ Step 6: Webcam opened successfully", flush=True)

WINDOW_NAME = "Exercise Detection & Feedback"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

print("✓ Step 7: Window created", flush=True)
print("\n★ APP STARTING - You should see the window now!", flush=True)
print("Press ESC to quit\n", flush=True)

is_fullscreen = False
paused = False
rep_count = 0
stage = "up"
time1 = 0
frame_count = 0

try:
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame", flush=True)
                break

            frame = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_pose = pose.process(img_rgb)
            results_hands = hands.process(img_rgb)

            if results_pose.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                landmarks = extract_landmarks(results_pose.pose_landmarks.landmark)
                norm_landmarks = preprocess_landmarks(landmarks)

                if model is not None:
                    try:
                        exercise_pred = model.predict([norm_landmarks])[0]
                    except Exception as e:
                        exercise_pred = "unknown"
                else:
                    exercise_pred = "no_model"

                lm = results_pose.pose_landmarks.landmark

                hip = [lm[24].x, lm[24].y, lm[24].z]
                knee = [lm[26].x, lm[26].y, lm[26].z]
                ankle = [lm[28].x, lm[28].y, lm[28].z]
                elbow = [lm[14].x, lm[14].y, lm[14].z]
                shoulder = [lm[12].x, lm[12].y, lm[12].z]
                wrist = [lm[16].x, lm[16].y, lm[16].z]

                knee_angle = calculate_angle(hip, knee, ankle)
                elbow_angle = calculate_angle(shoulder, elbow, wrist)

                angles = {"knee": knee_angle, "elbow": elbow_angle}

                rep_count, stage, feedback = get_feedback_and_count(exercise_pred, angles, rep_count, stage)

                info_lines = [
                    f'Exercise: {exercise_pred}',
                    f'Reps: {rep_count}',
                    f'Stage: {stage}',
                    f'Feedback: {feedback}',
                    f'Knee Angle: {int(knee_angle)}',
                    f'Elbow Angle: {int(elbow_angle)}',
                ]
                draw_info_panel(frame, info_lines, x=8, y=8, padding=10, bg_color=(12, 12, 12), alpha=0.55)

            if results_hands.multi_hand_landmarks:
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            time2 = time()
            fps = 1 / (time2 - time1 + 1e-6)
            time1 = time2
            cv2.putText(frame, f'FPS: {int(fps)}', (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

            cv2.imshow(WINDOW_NAME, frame)
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"✓ Running... Frame {frame_count}, FPS: {int(fps)}", flush=True)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # escape key to quit
            print("User pressed ESC - quitting", flush=True)
            break
        elif key == ord('t'):  # toggle camera
            current_camera = 1 - current_camera
            cap.release()
            cap = cv2.VideoCapture(current_camera)
        elif key == ord('p'):  # pause/resume detection
            paused = not paused
        elif key == ord('m'):  # toggle fullscreen
            is_fullscreen = not is_fullscreen
            prop = cv2.WND_PROP_FULLSCREEN
            cv2.setWindowProperty(WINDOW_NAME, prop, cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL)

except KeyboardInterrupt:
    print("Interrupted by user", flush=True)
except Exception as e:
    print(f"Error during execution: {e}", flush=True)
    import traceback
    traceback.print_exc()

print("Cleaning up...", flush=True)
cap.release()
cv2.destroyAllWindows()
print("✓ Done!", flush=True)
