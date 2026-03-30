import os
import numpy as np
import cv2
from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# IMPORTANT:
# Same class order as training notebook
CLASSES = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']

# Load model
model = load_model("final_video_action_model.keras")


def extract_frames(video_path, max_frames=20, img_size=96):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    if total_frames > 0:
        frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    else:
        frame_indices = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if not ret:
            frames.append(np.zeros((img_size, img_size, 3), dtype=np.float32))
            continue

        frame = cv2.resize(frame, (img_size, img_size))
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)

    cap.release()

    while len(frames) < max_frames:
        frames.append(np.zeros((img_size, img_size, 3), dtype=np.float32))

    return np.array(frames, dtype=np.float32)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    video_path = None

    if request.method == "POST":
        file = request.files.get("video")

        if file and file.filename != "":
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            # IMPORTANT:
            # Match notebook settings
            frames = extract_frames(filepath, max_frames=20, img_size=96)
            frames = np.expand_dims(frames, axis=0)

            pred = model.predict(frames, verbose=0)[0]
            class_index = int(np.argmax(pred))

            prediction = CLASSES[class_index]
            confidence = round(float(pred[class_index]) * 100, 2)

            video_path = "/uploads/" + file.filename

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        video_path=video_path
    )


if __name__ == "__main__":
    app.run(debug=True)