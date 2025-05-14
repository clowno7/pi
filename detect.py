
# detect.py - Refined Gender and Age Detection Script

import cv2
import argparse

def get_face_box(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes

def main(image_path=None):
    faceProto = "models/opencv_face_detector.pbtxt"
    faceModel = "models/opencv_face_detector_uint8.pb"
    ageProto = "models/age_deploy.prototxt"
    ageModel = "models/age_net.caffemodel"
    genderProto = "models/gender_deploy.prototxt"
    genderModel = "models/gender_net.caffemodel"

    MODEL_MEAN_VALUES = (78.426, 87.769, 114.896)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']

    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)

    video = cv2.VideoCapture(image_path if image_path else 0)
    padding = 20

    while cv2.waitKey(1) < 0:
        hasFrame, frame = video.read()
        if not hasFrame:
            break

        resultImg, faceBoxes = get_face_box(faceNet, frame)
        if not faceBoxes:
            print("No face detected")
            continue

        for faceBox in faceBoxes:
            face = frame[max(0, faceBox[1] - padding):
                         min(faceBox[3] + padding, frame.shape[0] - 1),
                         max(0, faceBox[0] - padding):
                         min(faceBox[2] + padding, frame.shape[1] - 1)]

            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                         MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            gender = genderList[genderNet.forward()[0].argmax()]

            ageNet.setInput(blob)
            age = ageList[ageNet.forward()[0].argmax()]

            label = f"{gender}, {age}"
            cv2.putText(resultImg, label, (faceBox[0], faceBox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            print(f"Detected: {label}")

        cv2.imshow("Age and Gender Detection", resultImg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help='Path to image or leave empty for webcam')
    args = parser.parse_args()
    main(args.image)
