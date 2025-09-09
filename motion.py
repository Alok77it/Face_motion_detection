"""
smart_attention.py
Single-file prototype for Smart Face Motion & Attention Detection.

Usage:
    python smart_attention.py
Press ESC to quit.

Outputs:
    - logs.csv written to current directory
    - saved frames for critical events in ./events/
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
VIDEO_SOURCE = 0  # 0 for default webcam; or "path/to/video.mp4"
LOG_CSV = "logs.csv"
EVENT_DIR = "events"
SAVE_EVENT_FRAMES = True

# thresholds
MOVE_THRESHOLD_PIX = 12       # pixels to consider as movement
INATTENTIVE_TIME_SEC = 4      # continuous seconds away to mark inattentive
DROWSY_EAR_THRESHOLD = 0.22   # eye aspect ratio threshold for closed eyes
DROWSY_CONSEC_FRAMES = 40     # frames of closed eyes to consider drowsy
BLINK_CONSEC_FRAMES = 2       # frames threshold to count blink

# smoothing
CENTER_HISTORY_LEN = 5

os.makedirs(EVENT_DIR, exist_ok=True)


# -----------------------------
# Mediapipe init
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# face mesh indices used (Mediapipe face mesh indexing)
# Left eye landmarks (approx)
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]   # common indexes for left eye region
RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373] # common indexes for right eye

# For head pose approximation choose nose tip and forehead-ish points
NOSE_TIP_IDX = 1
FOREHEAD_IDX = 10  # approximate small offset for vertical pose check

# -----------------------------
# Utils
# -----------------------------
def normalized_to_pixel_coords(norm_x, norm_y, image_width, image_height):
    x_px = min(int(norm_x * image_width), image_width - 1)
    y_px = min(int(norm_y * image_height), image_height - 1)
    return x_px, y_px

def compute_centroid(landmarks, image_w, image_h):
    xs = []
    ys = []
    for lm in landmarks:
        x_px, y_px = normalized_to_pixel_coords(lm.x, lm.y, image_w, image_h)
        xs.append(x_px)
        ys.append(y_px)
    return int(np.mean(xs)), int(np.mean(ys))

def eye_aspect_ratio(landmarks, eye_idx_list, image_w, image_h):
    # Use two vertical distances and one horizontal distance approximate EAR
    pts = []
    for idx in eye_idx_list:
        lm = landmarks[idx]
        x, y = normalized_to_pixel_coords(lm.x, lm.y, image_w, image_h)
        pts.append((x, y))
    # vertical distances (p2-p6 and p3-p5 in typical EAR)
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    # horizontal
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    if C == 0:
        return 0.0
    ear = (A + B) / (2.0 * C)
    return ear

def head_direction(landmarks, image_w, image_h):
    # crude method: compare nose x to face center to understand left/right
    nose = landmarks[NOSE_TIP_IDX]
    forehead = landmarks[FOREHEAD_IDX] if FOREHEAD_IDX < len(landmarks) else landmarks[NOSE_TIP_IDX]
    nx, ny = normalized_to_pixel_coords(nose.x, nose.y, image_w, image_h)
    fx, fy = normalized_to_pixel_coords(forehead.x, forehead.y, image_w, image_h)
    # horizontal offset
    dx = nx - fx
    # normalized dx by image width
    ndx = dx / image_w
    # thresholds are heuristic
    if ndx > 0.03:
        return "Right"
    elif ndx < -0.03:
        return "Left"
    else:
        # check vertical: nose below forehead => looking down
        if ny - fy > image_h * 0.02:
            return "Down"
        elif fy - ny > image_h * 0.02:
            return "Up"
        else:
            return "Center"

# -----------------------------
# Logging
# -----------------------------
cols = ["timestamp", "frame_idx", "face_conf", "cx", "cy", "dx", "dy",
        "direction_move", "head_direction", "left_EAR", "right_EAR",
        "avg_EAR", "blink", "drowsy", "inattentive"]
log_df = pd.DataFrame(columns=cols)

# -----------------------------
# Main loop
# -----------------------------
cap = cv2.VideoCapture(VIDEO_SOURCE)
frame_idx = 0
prev_center = None
center_history = deque(maxlen=CENTER_HISTORY_LEN)
last_face_seen_time = None
blink_counter = 0
drowsy_counter = 0
inattentive_flag = False

print("Starting. Press ESC to quit.")
start_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame; exiting.")
        break
    frame_idx += 1
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    timestamp = datetime.utcnow().isoformat()
    face_conf = 0.0
    cx = cy = -1
    dx = dy = 0
    move_dir = "None"
    head_dir = "Unknown"
    left_EAR = right_EAR = avg_EAR = 0.0
    blink = 0
    drowsy = 0

    if results.multi_face_landmarks:
        face_conf = 1.0
        landmarks = results.multi_face_landmarks[0].landmark

        # draw mesh
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=results.multi_face_landmarks[0],
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )

        # centroid / center of face from a subset of landmarks (use face oval indices)
        face_indices = [10, 152, 234, 454, 127, 356]  # sample face points
        cx, cy = compute_centroid([landmarks[i] for i in face_indices], w, h)
        center_history.append((cx, cy))

        if prev_center is None:
            prev_center = (cx, cy)
        dx = cx - prev_center[0]
        dy = cy - prev_center[1]

        # movement direction
        if abs(dx) > MOVE_THRESHOLD_PIX or abs(dy) > MOVE_THRESHOLD_PIX:
            if abs(dx) >= abs(dy):
                move_dir = "Right" if dx > 0 else "Left"
            else:
                move_dir = "Down" if dy > 0 else "Up"
        else:
            move_dir = "Stable"

        prev_center = (cx, cy)
        last_face_seen_time = time.time()

        # head direction estimation
        head_dir = head_direction(landmarks, w, h)

        # EAR
        left_EAR = eye_aspect_ratio(landmarks, LEFT_EYE_IDX, w, h)
        right_EAR = eye_aspect_ratio(landmarks, RIGHT_EYE_IDX, w, h)
        avg_EAR = (left_EAR + right_EAR) / 2.0

        # blink & drowsy detection
        if avg_EAR < DROWSY_EAR_THRESHOLD:
            drowsy_counter += 1
            blink_counter = 0
        else:
            if drowsy_counter >= BLINK_CONSEC_FRAMES:
                blink = 1
            drowsy_counter = 0

        if drowsy_counter >= DROWSY_CONSEC_FRAMES:
            drowsy = 1

        # draw center and status
        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
        cv2.putText(frame, f"Move:{move_dir}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, f"Head:{head_dir}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, f"EAR:{avg_EAR:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # Alerts
        if drowsy:
            cv2.putText(frame, "DROWSY! ALERT!", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
            if SAVE_EVENT_FRAMES:
                fname = os.path.join(EVENT_DIR, f"drowsy_{frame_idx}_{int(time.time())}.jpg")
                cv2.imwrite(fname, frame)
        # inattentive: head not center OR not looking at screen (heuristic)
        inattentive = (head_dir in ["Left", "Right", "Down", "Up"] and head_dir != "Center" and move_dir != "Stable")
        # also if not detected for a while we'll set inattentive below
        cv2.putText(frame, f"Inattentive:{inattentive}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    else:
        # no face detected
        if last_face_seen_time is None:
            last_face_seen_time = time.time()
        secs_since = time.time() - last_face_seen_time
        if secs_since > INATTENTIVE_TIME_SEC:
            inattentive_flag = True
            cv2.putText(frame, "No face detected: possibly away", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            if SAVE_EVENT_FRAMES:
                fname = os.path.join(EVENT_DIR, f"no_face_{frame_idx}_{int(time.time())}.jpg")
                cv2.imwrite(fname, frame)
        else:
            inattentive_flag = False

    # record log row
    log_row = {
        "timestamp": timestamp,
        "frame_idx": frame_idx,
        "face_conf": face_conf,
        "cx": cx, "cy": cy, "dx": dx, "dy": dy,
        "direction_move": move_dir,
        "head_direction": head_dir,
        "left_EAR": round(left_EAR, 3),
        "right_EAR": round(right_EAR, 3),
        "avg_EAR": round(avg_EAR, 3),
        "blink": blink,
        "drowsy": drowsy,
        "inattentive": bool(inattentive_flag or (last_face_seen_time and (time.time() - last_face_seen_time) > INATTENTIVE_TIME_SEC))
    }
    log_df = log_df.append(log_row, ignore_index=True)

    # show frame
    cv2.imshow("Smart Attention", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

    # periodically save logs to CSV
    if frame_idx % 200 == 0:
        log_df.to_csv(LOG_CSV, index=False)

# final save
log_df.to_csv(LOG_CSV, index=False)
cap.release()
cv2.destroyAllWindows()
print(f"Finished. Logs saved to {LOG_CSV} and event frames (if any) in {EVENT_DIR}")
