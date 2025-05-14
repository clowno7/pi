# monitor.py

import cv2
import time
from utils import get_face_box, is_unsafe_time
from config import AGE_MODEL, GENDER_MODEL, AGE_BUCKETS, GENDER_LIST
import numpy as np
import datetime
import os

print("[INFO] Starting monitor.py...")

# Load models
faceProto = "models/opencv_face_detector.pbtxt"
faceModel = "models/opencv_face_detector_uint8.pb"
faceNet = cv2.dnn.readNet(faceModel, faceProto)

ageNet = cv2.dnn.readNet(AGE_MODEL['model'], AGE_MODEL['proto'])
genderNet = cv2.dnn.readNet(GENDER_MODEL['model'], GENDER_MODEL['proto'])

# Try to open webcam
cap = cv2.VideoCapture(0)
use_webcam = cap.isOpened()

if not use_webcam:
    print("[WARNING] Webcam not accessible! Falling back to image.")
    fallback_image = "kid1.jpg"  # You can change this
    frame = cv2.imread(fallback_image)
    if frame is None:
        print(f"[ERROR] Failed to load fallback image: {fallback_image}")
        exit()
else:
    print("[INFO] Webcam accessed successfully.")

# Padding for face box
padding = 20

while True:
    if use_webcam:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame from webcam")
            break
    # Else we keep using the same fallback image

    _, bboxes = get_face_box(faceNet, frame)
    print(f"[DEBUG] Found {len(bboxes)} face(s)")
    for bbox in bboxes:
        face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
                     max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                     (78.426, 87.768, 114.895), swapRB=False)

        genderNet.setInput(blob)
        gender = GENDER_LIST[genderNet.forward().argmax()]

        ageNet.setInput(blob)
        age = AGE_BUCKETS[ageNet.forward().argmax()]

        print(f"[DEBUG] Detected: Gender={gender}, Age={age}")

        label = f"{gender}, {age}"
        cv2.putText(frame, label, (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                      (255, 0, 0), 2)

        # Call is_unsafe_time and it will print its log
        is_unsafe_time()

        if any(x in age for x in ['0-2', '4-6', '8-12']) and is_unsafe_time():
            print(f"[ALERT] Child detected at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            with open("logs/alerts.log", "a") as f:
                f.write(f"ALERT: Child detected at {datetime.datetime.now()}\n")

    cv2.imshow("Age-Gender Monitor", frame)
    key = cv2.waitKey(0 if not use_webcam else 1)
    if key == 27:  # ESC
        break

    # Add a small delay
    time.sleep(0.1)

if use_webcam:
    cap.release()
cv2.destroyAllWindows()
