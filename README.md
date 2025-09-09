1. Project Summary (one-line)

Smart Face Motion & Attention Detection — Realtime system that detects faces, tracks head motion, detects blinks/drowsiness, computes attention level, logs events, and triggers alerts.

2. Deliverables included here

Full working Python prototype (modular functions) — below.

requirements.txt

Run instructions and demo script.

Logging format and sample CSV column description.

Project report structure & slides outline.

Evaluation metrics and experiments to run.

Ideas for extra credit (emotion detection, Flask dashboard, multi-person).

3. Tech stack / Libraries

Python 3.9+

OpenCV (opencv-python)

Mediapipe (mediapipe) — face mesh & landmarks

NumPy, Pandas

Playsound or simple OS beep for alerts (we use OpenCV window beeps by default)

4. How it works (brief)

Use Mediapipe Face Mesh to detect facial landmarks (eyes, nose, face outline).

Compute face center and track its movement (centroid delta) between frames → motion direction and speed.

Eye Aspect Ratio (EAR) style measure (from landmarks) to detect blinks and prolonged eye closure → drowsiness.

Head pose approximation using key landmarks to detect left/right/up/down orientation → attention.

System logs events with timestamp, status, direction, EAR, head_pose, and saves frames on critical events.

Trigger audible & on-screen alerts when inattentive or drowsy.
