Human Activity Recognition using Deep Learning
📌 Project Overview

This project focuses on video classification using Deep Learning.
It predicts human activities from input video clips.

🚀 Features
Detects activities:
Boxing
Handclapping
Handwaving
Jogging
Running
Walking
Uses CNN + LSTM / MobileNet
Frame extraction from video
Displays predicted activity
🛠️ Technologies Used
Python
TensorFlow / Keras
OpenCV
NumPy
Flask (for UI)
📂 Project Structure
video_project/
│
├── model/
│   └── video_model.keras
│
├── static/
│   ├── css/
│   ├── videos/
│
├── templates/
│   └── index.html
│
├── app.py
├── requirements.txt
└── README.md
⚙️ How It Works
Upload video file
Extract frames using OpenCV
Preprocess frames
Pass sequence to model
Predict activity
▶️ How to Run
git clone <repo-link>
cd video_project
pip install -r requirements.txt
python app.py

Open: http://127.0.0.1:5000/

📊 Dataset
KTH Action Recognition Dataset
📈 Results
Accuracy: ~79–85%
📌 Future Improvements
Increase accuracy to 95%+
Real-time activity detection
Add more activity classes
Improve UI/UX
👩‍💻 Team Members
Tejaswini Soni
Suhani Khare
Shrishty Alanse
📢 Acknowledgement

We thank our mentors and SGSITS, Indore for their guidance.
