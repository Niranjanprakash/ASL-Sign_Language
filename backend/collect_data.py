"""
collect_data.py - Webcam-based data collection for ASL handshapes.
Press a letter key (A-Z) to set the active class.
Press SPACE to capture a sample.
Press ESC to quit.

Samples saved as .npy files under: data/<LABEL>/<timestamp>.npy
Each .npy file contains 63 raw landmark floats (21 landmarks × [x,y,z]).
"""

import os
import sys
import time
import cv2
import numpy as np

# MediaPipe Tasks API
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

sys.path.insert(0, os.path.dirname(__file__))
from utils import LABELS

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_FILE = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")


def create_detector():
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_FILE)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=mp_vision.RunningMode.VIDEO,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def draw_landmarks(frame, landmarks, h, w):
    from utils import HAND_CONNECTIONS
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for connection in HAND_CONNECTIONS:
        cv2.line(frame, pts[connection[0]], pts[connection[1]], (255, 150, 0), 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (0, 255, 100), -1)
    return pts


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    detector = create_detector()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    active_label = "A"
    frame_ts = 0
    print("=== ASL Data Collector ===")
    print("Press A-Z to select class | SPACE to capture | ESC to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        frame_ts += 33
        result = detector.detect_for_video(mp_img, frame_ts)

        raw_landmarks = None
        if result.hand_landmarks:
            lm = result.hand_landmarks[0]
            draw_landmarks(frame, lm, h, w)
            raw_landmarks = [coord for pt in lm for coord in (pt.x, pt.y, pt.z)]

        # Count existing samples
        label_dir = os.path.join(DATA_DIR, active_label)
        count = len(os.listdir(label_dir)) if os.path.isdir(label_dir) else 0

        # HUD
        cv2.rectangle(frame, (0, 0), (w, 50), (20, 20, 40), -1)
        cv2.putText(frame, f"Class: {active_label}  Samples: {count}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 180), 2)
        status = "Hand Detected" if raw_landmarks else "No Hand"
        color = (0, 255, 100) if raw_landmarks else (0, 80, 220)
        cv2.putText(frame, status, (w - 220, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, "SPACE=Capture  ESC=Quit",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        cv2.imshow("ASL Data Collector", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        elif ord('a') <= key <= ord('z'):
            active_label = chr(key).upper()
            print(f"[CLASS] Switched to: {active_label}")
        elif ord('A') <= key <= ord('Z'):
            active_label = chr(key)
            print(f"[CLASS] Switched to: {active_label}")
        elif key == ord(' '):
            if raw_landmarks:
                os.makedirs(label_dir, exist_ok=True)
                ts = str(int(time.time() * 1000))
                fpath = os.path.join(label_dir, f"{ts}.npy")
                np.save(fpath, np.array(raw_landmarks, dtype=np.float32))
                print(f"  ✓ Saved {fpath}  (total: {count + 1})")
                cv2.rectangle(frame, (0, 0), (w, h), (0, 255, 100), 4)
                cv2.imshow("ASL Data Collector", frame)
                cv2.waitKey(80)
            else:
                print("  ✗ No hand detected – move your hand into frame")

    cap.release()
    cv2.destroyAllWindows()
    print("Data collection complete.")


if __name__ == "__main__":
    main()
