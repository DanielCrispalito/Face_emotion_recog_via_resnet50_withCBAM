from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

# ===== IMPORT MODEL =====
from model import ResNet50_LightCBAM

app = Flask(__name__)

# ===== PATH =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===== DEVICE =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== LOAD MODEL =====
model = ResNet50_LightCBAM(num_classes=7)
model.load_state_dict(
    torch.load(
        os.path.join(BASE_DIR, "best_resnet50_cbam.pth"),
        map_location=device
    )
)
model.to(device)
model.eval()

# ===== HAAR CASCADE =====
face_cascade = cv2.CascadeClassifier(
    os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
)

# ===== LABEL =====
emotion_labels = [
    'Surprise',
    'Fear',
    'Disgust',
    'Happiness',
    'Sadness',
    'Anger',
    'Neutral'
]


# ===== IMAGE TRANSFORM (SAMA DENGAN TRAINING) =====
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ===== ROUTES =====
@app.route("/")
def index():
    return render_template("dashboard.html", labels=emotion_labels)


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    img_bytes = np.frombuffer(file.read(), np.uint8)

    # Decode image (BGR)
    frame = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Invalid image"}), 400

    # Grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ===== FACE DETECTION =====
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    if len(faces) == 0:
        return jsonify({"error": "No face detected"}), 400

    # Ambil wajah pertama
    x, y, w, h = faces[0]
    face = frame[y:y+h, x:x+w]

    # BGR -> RGB
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    # ===== PREPROCESS =====
    input_tensor = transform(face).unsqueeze(0).to(device)

    # ===== INFERENCE =====
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)[0]

    emotion_idx = torch.argmax(probs).item()

    return jsonify({
        "emotion": emotion_labels[emotion_idx],
        "confidence": float(probs[emotion_idx]),
        "scores": probs.cpu().numpy().tolist()
    })


if __name__ == "__main__":
    app.run(debug=True)
