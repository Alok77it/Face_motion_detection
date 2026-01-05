"""
smart_attention.py
Smart Face Motion & Attention Detection

Requirements:
- Python 3.10 or 3.11 ONLY
- pip install mediapipe opencv-python numpy pandas

Run:
python smart_attention.py

Press ESC to quit.

Outputs:
- logs.csv
- ./events/ (saved alert frames)
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import time
from collections import deque
from datetime import datetime

# -----------------------------
# CONFIG
# -----------------------------
VIDEO_SOURCE = 0
LOG_CSV = "logs.csv"
EVENT_DIR = "events"
SAVE_EVENT_FRAMES = True

MOVE_THRESHOLD_PIX = 12
INATTENTIVE_TIME_SEC = 4
DROWSY_EAR_THRESHOLD = 0.22
DROWSY_CONSEC_FRAMES = 40
BLINK_CONSEC_FRAMES = 2

CENTER_HISTORY_LEN = 5

os.makedirs(EVENT_DIR, exist_ok=True)

# -----------------------------
# MEDIAPIPE INIT
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]

NOSE_TIP_IDX = 1
FOREHEAD_IDX = 10

# -----------------------------
# UTILS
# -----------------------------
def norm_to_px(lm, w, h):
    return int(lm.x * w), int(lm.y * h)

def compute_centroid(landmarks, idxs, w, h):
    xs, ys = [], []
    for i in idxs:
        x, y = norm_to_px(landmarks[i], w, h)
        xs.append(x)
        ys.append(y)
    return int(np.mean(xs)), int(np.mean(ys))

def eye_aspect_ratio(landmarks, idxs, w, h):
    pts = [norm_to_px(landmarks[i], w, h) for i in idxs]
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return 0.0 if C == 0 else (A + B) / (2.0 * C)

def head_direction(landmarks, w, h):
    nose = landmarks[NOSE_TIP_IDX]
    forehead = landmarks[FOREHEAD_IDX]
    nx, ny = norm_to_px(nose, w, h)
    fx, fy = norm_to_px(forehead, w, h)

    dx = (nx - fx) / w
    dy = (ny - fy) / h

    if dx > 0.03:
        return "Right"
    if dx < -0.03:
        return "Left"
    if dy > 0.04:
        return "Down"
    if dy < -0.04:
        return "Up"
    return "Center"

# -----------------------------
# LOGGING
# -----------------------------
columns = [
    "timestamp", "frame",
    "cx", "cy", "dx", "dy",
    "move", "head",
    "left_EAR", "right_EAR", "avg_EAR",
    "blink", "drowsy", "inattentive"
]
log_rows = []

# -----------------------------
# MAIN LOOP
# -----------------------------
cap = cv2.VideoCapture(VIDEO_SOURCE)
frame_idx = 0
prev_center = None
last_face_time = time.time()
drowsy_counter = 0
blink_flag = 0

print("Started. Press ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    timestamp = datetime.utcnow().isoformat()
    inattentive = False
    blink = 0
    drowsy = 0
    move = "None"
    head = "Unknown"
    cx = cy = dx = dy = -1
    left_EAR = right_EAR = avg_EAR = 0.0

    if results.multi_face_landmarks:
        last_face_time = time.time()
        lm = results.multi_face_landmarks[0].landmark

        mp_drawing.draw_landmarks(
            frame,
            results.multi_face_landmarks[0],
            mp_face_mesh.FACEMESH_TESSELATION,
            None,
            mp_styles.get_default_face_mesh_tesselation_style()
        )

        face_pts = [10, 152, 234, 454, 127, 356]
        cx, cy = compute_centroid(lm, face_pts, w, h)

        if prev_center:
            dx = cx - prev_center[0]
            dy = cy - prev_center[1]
        prev_center = (cx, cy)

        if abs(dx) > MOVE_THRESHOLD_PIX or abs(dy) > MOVE_THRESHOLD_PIX:
            move = "Right" if abs(dx) > abs(dy) and dx > 0 else \
                   "Left" if abs(dx) > abs(dy) else \
                   "Down" if dy > 0 else "Up"
        else:
            move = "Stable"

        head = head_direction(lm, w, h)

        left_EAR = eye_aspect_ratio(lm, LEFT_EYE_IDX, w, h)
        right_EAR = eye_aspect_ratio(lm, RIGHT_EYE_IDX, w, h)
        avg_EAR = (left_EAR + right_EAR) / 2

        if avg_EAR < DROWSY_EAR_THRESHOLD:
            drowsy_counter += 1
        else:
            if drowsy_counter >= BLINK_CONSEC_FRAMES:
                blink = 1
            drowsy_counter = 0

        if drowsy_counter >= DROWSY_CONSEC_FRAMES:
            drowsy = 1
            if SAVE_EVENT_FRAMES:
                cv2.imwrite(
                    f"{EVENT_DIR}/drowsy_{frame_idx}.jpg", frame
                )

        inattentive = head != "Center"

        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

    else:
        if time.time() - last_face_time > INATTENTIVE_TIME_SEC:
            inattentive = True
            cv2.putText(
                frame,
                "NO FACE DETECTED",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )

    # UI
    cv2.putText(frame, f"Move: {move}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, f"Head: {head}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, f"EAR: {avg_EAR:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    if drowsy:
        cv2.putText(frame, "DROWSY ALERT!", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    cv2.imshow("Smart Attention", frame)

    log_rows.append([
        timestamp, frame_idx,
        cx, cy, dx, dy,
        move, head,
        round(left_EAR, 3), round(right_EAR, 3), round(avg_EAR, 3),
        blink, drowsy, inattentive
    ])

    if cv2.waitKey(1) & 0xFF == 27:
        break

# -----------------------------
# CLEANUP
# -----------------------------
cap.release()
cv2.destroyAllWindows()

df = pd.DataFrame(log_rows, columns=columns)
df.to_csv(LOG_CSV, index=False)

print(f"Finished. Logs saved to {LOG_CSV}")
