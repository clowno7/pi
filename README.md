
# Gender and Age Detection Using Computer Vision

## 🔍 Overview

This project implements a computer vision system that detects human faces in real-time and predicts their gender and age group using pre-trained deep learning models. It is useful in applications such as demographic analysis, targeted advertising, and intelligent surveillance systems.

## 📌 Features

- Real-time face detection using OpenCV
- Gender classification (Male/Female)
- Age estimation in ranges (e.g., 0–2, 4–6, 8–12, etc.)
- Pre-trained deep learning models for fast and accurate inference
- Easily deployable on laptops or embedded systems like Raspberry Pi

## 🧠 Technology Stack

- **Language**: Python 3.x  
- **Libraries**: OpenCV, TensorFlow/Keras, NumPy  
- **Models**: Pre-trained CNN models for age and gender prediction

## 🏗️ Project Structure

```
detect/
│
├── age_deploy.prototxt       # Model architecture for age
├── age_net.caffemodel        # Pre-trained age model
├── gender_deploy.prototxt    # Model architecture for gender
├── gender_net.caffemodel     # Pre-trained gender model
├── haarcascade_frontalface.xml # OpenCV face detector
├── detect.py                 # Main Python script
└── output.png                # Example output
```

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/clowno7/Gender-and-Age-detection.git
cd gender-age-detection/detect
```

### 2. Install Dependencies

```bash
pip install opencv-python tensorflow numpy
```

### 3. Run the Program

```bash
python detect.py
```

The webcam will open and display real-time predictions of gender and age over detected faces.

## 🖼️ Example Output

![Output Example](output.png)

## 🧪 How It Works

1. The webcam captures real-time video.
2. OpenCV's Haarcascade detects faces in the frame.
3. Detected faces are cropped and resized for model input.
4. Gender and age models predict respective outputs.
5. The predictions are displayed on the video frame.

## 🎯 Applications

- Smart Advertising Screens
- Retail Demographic Analysis
- Public Security Systems
- Human-Robot Interaction

## 🔮 Future Improvements

- Improve accuracy with better and diverse datasets
- Include emotion detection
- Optimize for edge computing devices
- Support multiple face tracking

## 📚 References

- [OpenCV](https://opencv.org/)
- [Keras](https://keras.io/)
- [Pre-trained Models Source](https://github.com/yu4u/age-gender-estimation)

## 📝 License

This project is licensed under the MIT License.

