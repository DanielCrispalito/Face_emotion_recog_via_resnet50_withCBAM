# Face Emotion Recognition App 🎭

A web-based **Face Emotion Recognition (FER)** application that detects human facial emotions in real-time using **Haar Cascade** for face detection and a **Resnet-50 with CBAM** model trained on the **RAF-DB dataset**.

---

## 📌 Features

- Real-time face detection using **OpenCV Haar Cascade**
- Emotion classification using **Resnet-50 with CBAM**
- Displays:
  - Detected face
  - Predicted emotion
  - Confidence level
- Web interface built with **Flask**
- Supports live webcam input

---

## 😃 Detected Emotions

The model is trained to classify the following emotions:

- Angry
- Disgust
- Fear
- Happy
- Sad
- Neutral
- Surprise
  > ⚠️ Note: The model still has difficulty recognizing the **Surprise** emotion accurately.

---

## 🧠 Model Information

- Dataset: **RAF-DB**
- Model type: **ResNet-50 with CBAM**
- Framework: **PyTorch**
- Model format: `.pth`
- Input: RGB image (100x100)
- Output: Emotion label + confidence score
- Accuracy: ~ **80%**

---

## 🛠️ Tech Stack

- flask==3.0.0
- torch>=2.0.0
- torchvision>=0.15.0
- opencv-python>=4.8.0
- numpy>=1.23.0
- Pillow>=9.5.0

---

## 🚀 How to Run Locally

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/face_emotion_recog_apps.git
cd face_emotion_recog_realtime

python -m venv venv
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate     # Windows

pip install -r requirements.txt

python app.py

http://127.0.0.1:5000


```
