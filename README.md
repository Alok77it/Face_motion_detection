# Smart Face Motion & Attention Detection

## Overview

**Smart Face Motion & Attention Detection** is a real-time system developed in Python that leverages cutting-edge computer vision technologies to track head motion, detect facial landmarks, and monitor user attention levels. This system detects faces, tracks head movement, computes attention levels, detects drowsiness through eye blink detection, and triggers real-time alerts for attention and safety. 

By integrating **Mediapipe** for facial landmark detection and **OpenCV** for real-time video processing, the system provides an intuitive and actionable approach to detecting drowsiness and inattention.

---

## Key Features

- **Face Detection**: Uses Mediapipe's face mesh to detect and track facial landmarks in real-time.
- **Head Motion Tracking**: Tracks the centroid of the face between frames to compute motion speed and direction.
- **Drowsiness Detection**: Detects blinking and prolonged eye closure using the Eye Aspect Ratio (EAR).
- **Attention Level Monitoring**: Computes the user's attention level based on head pose orientation (left, right, up, down).
- **Real-Time Alerts**: Triggers both visual and audible alerts when the user shows signs of drowsiness or inattention.
- **Event Logging**: Logs critical events with timestamps, including detected attention levels, eye closure, and movement direction.
  
---

## Tech Stack & Libraries

This project utilizes the following technologies and libraries:

- **Python 3.9+**: Core language for implementing the system.
- **OpenCV** (`opencv-python`): For video processing, face detection, and displaying alerts.
- **Mediapipe** (`mediapipe`): For detecting facial landmarks, including eyes, nose, and face outline.
- **NumPy** (`numpy`): For handling mathematical operations, arrays, and calculations.
- **Pandas** (`pandas`): For logging events and storing them in CSV format.
- **Playsound/OS Beep**: For triggering audible alerts (beeps) when critical events occur.

---

## Setup Instructions

To get started with this project, follow the steps below to set up the environment, install dependencies, and run the demo script.

### 1. Clone the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/your-username/smart-face-motion-attention-detection.git
cd smart-face-motion-attention-detection
