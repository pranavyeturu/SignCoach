# SignCoach AI

Realtime ASL fingerspelling tutor prototype.

SignCoach AI is a local computer vision app for learning ASL fingerspelling. The Phase 1 build focuses on the realtime webcam and hand-landmark foundation needed before training and serving an ASL letter classifier.

## Current Capabilities

- Opens the webcam in the browser
- Sends compressed frames to a local FastAPI backend over WebSocket
- Runs MediaPipe Hands on each frame
- Detects whether a hand is visible
- Returns 21 hand landmarks for a detected hand
- Draws a landmark skeleton and bounding box over the video feed
- Shows basic feedback states:
  - camera off
  - backend offline
  - hand detected
  - no hand detected
  - move closer to the camera
- Reads the local ASL dataset folder and displays class/image counts

## Not Built Yet

- ASL letter classification
- Model training scripts
- Prediction smoothing
- Practice mode scoring
- Quiz mode
- Progress dashboard

Those belong to Phase 2 and Phase 3.

## Dataset

The dataset is intentionally not committed to git because it is large. Place the ASL Alphabet dataset here:

```text
archive/asl_alphabet_train/asl_alphabet_train/
```

Expected structure:

```text
archive/asl_alphabet_train/asl_alphabet_train/
  A/
  B/
  C/
  ...
  Z/
  del/
  nothing/
  space/
```

The current local dataset contains 29 classes and 87,000 images.

## Requirements

- Python 3.10-3.12
- Node.js 20+
- npm
- Webcam-enabled browser

MediaPipe may not install or run correctly on newer Python versions such as Python 3.14. Use a compatible Python interpreter when creating the backend virtual environment.

## Start The Backend

From the repo root:

```bash
cd backend
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

If `backend/.venv` already exists, start from:

```bash
cd backend
source .venv/bin/activate
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Backend health check:

```text
http://127.0.0.1:8000/api/health
```

## Start The Frontend

In a second terminal, from the repo root:

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

Click `Start Camera`, allow webcam access, and move your hand into frame. The app should draw landmarks over your hand when detection succeeds.

## Validation

Checks used for the Phase 1 build:

```bash
cd backend
.venv/bin/python -m compileall app
```

```bash
cd frontend
npm run build
npm audit
```

The backend was also tested by sending a real dataset image to `/api/detect`; it returned `handDetected: true` with 21 landmarks.

## Privacy

Webcam frames are sent only from the browser to the local backend running on `localhost`. Frames are not saved by default and are not uploaded to a cloud service.
