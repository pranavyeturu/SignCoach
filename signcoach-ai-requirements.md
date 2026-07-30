# Requirements Document: SignCoach AI

## 1. Project Summary

SignCoach AI is a Python-based realtime computer vision application for learning ASL fingerspelling. It uses a webcam to detect hand landmarks, classify static ASL alphabet signs, compare the user's sign against a target letter, and provide realtime feedback, scoring, and progress tracking.

The recommended implementation is a local web app with a Python CV/ML backend and a lightweight browser frontend. The webcam runs in the browser, frames are sent to the Python backend for inference, and the frontend renders the realtime teaching interface.

## 2. Recommended Stack

### Core Language

- Python 3.10 or 3.11

### App Architecture

- FastAPI backend
- React + Vite frontend
- WebSocket connection for realtime frame inference

Why:

- FastAPI keeps the ML/CV logic in Python.
- React gives you a real product-like interface instead of a notebook/dashboard feel.
- Browser webcam access is more reliable through `getUserMedia`.
- WebSockets are simple enough for realtime prediction without full WebRTC complexity.
- This structure can later evolve into a production app.

### Computer Vision

- OpenCV
- MediaPipe

Why:

- OpenCV handles webcam frames and image processing.
- MediaPipe Hands gives realtime 21-point hand landmarks.
- Landmark-based classification is faster and more reliable than training a CNN from scratch.

### Machine Learning

- scikit-learn

Recommended MVP model:

- Random Forest Classifier

Backup model:

- Support Vector Machine

Why:

- Fast training
- Easy saving/loading with `joblib`
- Works well on tabular landmark features
- Easier to debug than a neural network under a tight deadline

### Data Processing

- NumPy
- Pandas

### Frontend

- React
- Vite
- TypeScript optional, JavaScript acceptable for speed
- CSS modules or plain CSS

Use the frontend for:

- Webcam capture
- Practice UI
- Quiz UI
- Live prediction display
- Progress panel
- End-session summary

### Visualization

- Recharts for frontend charts
- Matplotlib or Seaborn for offline confusion matrix generation

### Model Persistence

- joblib

### Optional

- TensorFlow or PyTorch only if there is extra time.
- FastAPI only if splitting the app into backend/frontend later.

## 3. Python Dependencies

Backend Python dependencies:

```txt
fastapi
uvicorn
websockets
opencv-python
mediapipe
numpy
pandas
scikit-learn
joblib
matplotlib
seaborn
python-multipart
pydantic
```

Frontend dependencies:

```txt
react
vite
recharts
lucide-react
```

Optional backend dependencies:

```txt
tensorflow
torch
torchvision
```

## 4. Suggested `requirements.txt`

```txt
fastapi==0.112.0
uvicorn==0.30.5
websockets==12.0
opencv-python==4.10.0.84
mediapipe==0.10.14
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.5.1
joblib==1.4.2
matplotlib==3.9.1
seaborn==0.13.2
python-multipart==0.0.9
pydantic==2.8.2
```

If dependency conflicts happen, loosen versions:

```txt
fastapi
uvicorn
websockets
opencv-python
mediapipe
numpy
pandas
scikit-learn
joblib
matplotlib
seaborn
python-multipart
pydantic
```

## 5. Dataset Requirements

### Recommended Dataset

Use an ASL alphabet image dataset with folders per class.

Expected structure:

```text
dataset/
  A/
    image1.jpg
    image2.jpg
  B/
    image1.jpg
  C/
  ...
```

Recommended classes for MVP:

- A-Z static letters

Optional classes:

- space
- delete
- nothing

### Important Dataset Constraint

Letters `J` and `Z` are motion-based signs. For a static image MVP, either:

- exclude `J` and `Z`, or
- include them only as experimental classes with reduced confidence expectations.

## 6. System Requirements

### Hardware

- Laptop or desktop with webcam
- 8 GB RAM minimum
- No GPU required for MVP

### Software

- Python 3.10+
- pip
- Virtual environment recommended
- Browser access for Streamlit app

### Runtime

- App should run locally.
- Webcam frames should remain local.
- No login required.
- No cloud storage required.

## 7. Repository Structure

Recommended structure:

```text
signcoach-ai/
  backend/
    app/
      main.py
      config.py
      landmarks.py
      features.py
      prediction.py
      feedback.py
      scoring.py
      schemas.py
    scripts/
      extract_landmarks.py
      train_model.py
      evaluate_model.py
    requirements.txt
  frontend/
    index.html
    package.json
    src/
      App.jsx
      main.jsx
      api.js
      components/
        CameraView.jsx
        PracticePanel.jsx
        QuizPanel.jsx
        ProgressPanel.jsx
        ModeTabs.jsx
      styles.css
  README.md
  data/
    raw/
    processed/
  models/
    sign_classifier.pkl
    label_encoder.pkl
    metrics.json
    confusion_matrix.png
```

## 8. Core Modules

### `landmarks.py`

Responsible for:

- Initializing MediaPipe Hands
- Detecting hand landmarks
- Returning 21 hand landmarks per frame
- Returning handedness if available

### `features.py`

Responsible for:

- Normalizing landmarks
- Converting landmarks into feature vectors
- Handling missing landmarks
- Optional distance and angle features

Recommended base feature vector:

```text
21 landmarks x 3 coordinates = 63 features
```

### `prediction.py`

Responsible for:

- Loading trained model
- Loading label encoder
- Running prediction
- Returning top class and confidence
- Returning top 3 classes if possible

### `feedback.py`

Responsible for:

- Creating user-facing feedback messages
- Handling no-hand state
- Handling low-confidence state
- Handling wrong-sign state
- Handling correct-sign state

### `scoring.py`

Responsible for:

- Confidence score
- Stability score
- Framing score
- Final attempt score
- Hold-to-confirm logic

### `session_state.py`

Responsible for:

- Attempts
- Correct count
- Incorrect count
- Current streak
- Best streak
- Per-letter accuracy
- Session duration

### `main.py`

Responsible for:

- FastAPI app initialization
- Health check endpoint
- Prediction endpoint
- WebSocket prediction endpoint
- Loading the model once at startup

### Frontend Components

Responsible for:

- Camera capture using browser `getUserMedia`
- Sending frames to backend
- Showing live prediction
- Showing target letter
- Showing teacher feedback
- Showing quiz/progress state

## 9. Functional Requirements

### Webcam

- App must request webcam access.
- App must show live video feed.
- App must process webcam frames in realtime.
- App must show a clear message when camera is unavailable.

### Hand Detection

- App must detect at least one hand in frame.
- App must draw hand landmarks on the video feed.
- App must show "No hand detected" when landmarks are missing.

### Sign Classification

- App must classify the current hand pose into a supported ASL letter.
- App must show prediction confidence.
- App must use a confidence threshold before confirming a prediction.
- App must smooth predictions across recent frames to reduce flicker.

### Practice Mode

- User must be able to select a target letter.
- App must compare prediction against target letter.
- App must show correct/incorrect status.
- App must require the correct prediction to be held briefly before counting success.

### Quiz Mode

- App must generate random target letters.
- App must track quiz score.
- App must move to the next letter after successful hold confirmation.

### Freestyle Mode

- App must predict any supported sign without a target.
- App must show top prediction and confidence.

### Dashboard

- App must show:
  - total attempts
  - correct attempts
  - incorrect attempts
  - accuracy
  - current streak
  - best streak
  - hardest letters
  - per-letter accuracy

## 10. Non-Functional Requirements

- Minimum realtime speed: 10 FPS.
- Target prediction latency: under 500 ms.
- App should work without internet after dependencies and model are installed.
- App should run on CPU.
- App should not store raw webcam video by default.
- Frontend should keep UI responsive while webcam is running.
- App should fail gracefully when no hand is visible.

## 11. ML Pipeline Requirements

### Landmark Extraction

The extraction script must:

1. Load images from dataset folders.
2. Run MediaPipe Hands on each image.
3. Skip images where no hand is detected.
4. Normalize landmarks.
5. Save features and labels to a processed file.

Recommended output:

```text
data/processed/landmarks.csv
```

### Training

The training script must:

1. Load processed landmarks.
2. Split train/test data.
3. Train classifier.
4. Evaluate accuracy.
5. Save model.
6. Save label encoder.
7. Save metrics.
8. Save confusion matrix.

### Evaluation

Minimum evaluation output:

- Accuracy
- Classification report
- Confusion matrix
- Per-class accuracy

## 12. Realtime Prediction Requirements

For each webcam frame:

1. Convert frame to RGB.
2. Run MediaPipe hand landmark detection.
3. If no hand is detected, return no-hand state.
4. Extract normalized features.
5. Run classifier prediction.
6. Apply confidence threshold.
7. Add prediction to smoothing window.
8. Compute final stable prediction.
9. Compare stable prediction with target letter if in teacher/quiz mode.
10. Update feedback and session metrics.

## 13. Prediction Smoothing Requirements

Use a rolling window of recent predictions.

Recommended defaults:

```text
prediction_window = 10 frames
confidence_threshold = 0.70
hold_duration_seconds = 1.0
```

Confirmation rules:

- A sign is confirmed only when the same letter is predicted for the hold duration.
- Predictions below confidence threshold should show "uncertain."
- If no hand is detected, reset hold progress.

## 14. Feedback Requirements

Initial feedback rules:

| Condition | Feedback |
|---|---|
| No hand detected | Move your hand into frame. |
| Hand too small | Move closer to the camera. |
| Low confidence | Hold steady and face your palm toward the camera. |
| Wrong sign | Detected X. Target is Y. |
| Correct but not held | Correct. Hold for one second. |
| Correct and held | Nice. Sign confirmed. |

## 15. Scoring Requirements

Attempt score:

```text
score = confidence_score * 0.70
      + stability_score * 0.15
      + framing_score * 0.15
```

Where:

- `confidence_score` comes from model probability.
- `stability_score` measures landmark movement across recent frames.
- `framing_score` measures whether the hand is centered and large enough.

## 16. Privacy Requirements

- Webcam frames should not be saved by default.
- No video recording should occur unless explicitly added and disclosed.
- User progress can be stored locally.
- The README must state that webcam processing is local.
- If any cloud model/API is added later, the app must clearly disclose what data is sent.

## 17. Performance Requirements

Target:

- 10+ FPS on normal laptop CPU
- Under 500 ms prediction latency
- Streamlit UI remains responsive
- Model loads once at app startup
- MediaPipe Hands object is reused instead of recreated per frame.
- Browser should send compressed frames at a controlled rate, such as 8-12 FPS.

## 18. Acceptance Criteria

The MVP is complete when:

- User can launch the backend and frontend locally.
- User can start webcam.
- App detects hand landmarks.
- App predicts supported ASL letters.
- App shows confidence.
- User can select a target letter.
- App gives correct/incorrect feedback.
- App requires hold-to-confirm before counting success.
- Dashboard shows session metrics.
- README explains setup and demo flow.

## 19. Build Timeline

### Day 1: Foundation

- Create repo structure
- Set up virtual environment
- Install dependencies
- Build FastAPI backend shell
- Build React frontend shell
- Add browser webcam feed
- Add MediaPipe hand detection
- Draw landmarks

### Day 2: Model

- Download ASL alphabet dataset
- Extract hand landmarks from images
- Train Random Forest or SVM
- Save model and label encoder
- Evaluate with confusion matrix
- Load model in backend

### Day 3: Realtime Teacher

- Add realtime prediction
- Add prediction smoothing
- Add target-letter practice
- Add quiz mode
- Add scoring
- Add progress dashboard
- Test demo letters
- Write README

## 20. Recommended Commands

Create virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Extract landmarks:

```bash
python scripts/extract_landmarks.py --input data/raw --output data/processed/landmarks.csv
```

Train model:

```bash
python scripts/train_model.py --input data/processed/landmarks.csv --output models
```

Run backend:

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

Run frontend:

```bash
cd frontend
npm install
npm run dev
```

## 21. Recommended MVP Shortcut

If dataset training takes too long, build a fallback version:

- Use MediaPipe landmarks.
- Support 5-8 manually tested letters.
- Collect 30-50 samples per letter from your webcam.
- Train on your own samples.
- Demo only those reliable letters.

Recommended demo letters:

- A
- B
- C
- L
- O
- V
- W
- Y

Avoid for first demo:

- M
- N
- T
- E
- S
- J
- Z

## 22. Final Recommendation

Use Python, FastAPI, OpenCV, MediaPipe, scikit-learn, and a React/Vite frontend.

Build the system around MediaPipe hand landmarks and a simple classifier. The strongest MVP is not "perfect ASL translation." It is a realtime ASL teacher that detects signs, provides feedback, and tracks progress.
