import cv2
import datetime
import config

def get_face_box(net, frame):
    frameOpencvDnn = frame.copy()
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300),
                                 [104, 117, 123], swapRB=False, crop=False)
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > config.CONFIDENCE_THRESHOLD:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2),
                          (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, faceBoxes

def is_unsafe_time():
    print("is_unsafe_time called")
    current_hour = datetime.datetime.now().hour
    unsafe_start, unsafe_end = config.UNSAFE_HOURS

    if unsafe_start < unsafe_end: # e.g., 10 PM to 5 AM
        return current_hour >= unsafe_start or current_hour < unsafe_end
    else: # e.g., 5 AM to 10 PM (shouldn't be the case based on config, but for robustness)
        return current_hour >= unsafe_start and current_hour < unsafe_end
