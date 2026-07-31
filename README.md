# SignCoach AI

A local, realtime ASL fingerspelling tutor with webcam landmarks, trained-letter recognition,
practice and quiz modes, hold-to-confirm feedback, scoring, and a browser-local progress dashboard.

## What is included

- MediaPipe hand detection with a landmark overlay and framing feedback
- Landmark feature extraction and Random Forest model training scripts
- Smoothed top prediction, confidence, top-three results, and one-second confirmation
- Practice mode with an A-Z selector
- Quiz mode with randomized prompts and skip/incorrect scoring
- Freestyle recognition
- Session accuracy, streaks, per-letter accuracy, and hardest-letter ordering
- Progress stored only in the browser

J and Z are motion-based ASL letters. A classifier trained on still images may recognize dataset
poses for them, but reliable motion recognition is outside this static-image MVP.

## Laptop safety and requirements

Running the app is safe on an ordinary laptop. It processes frames locally and does not record or
upload webcam video. Expect moderate CPU use while the camera is active. Stop the camera when you
are finished to release the webcam and CPU.

Recommended:

- Python 3.10-3.12 (MediaPipe is not expected to work with Python 3.14)
- Node.js 20+
- 8 GB RAM (4 GB may work but training will be slower)
- About 1 GB free for dependencies, plus roughly 3-5 GB for the full 87,000-image dataset

Training is the heavy step: it can run for tens of minutes, make the fan spin, and temporarily use
several CPU cores. It should not damage the laptop; keep it plugged in and ensure the vents are
clear. Use `--max-per-class 500` first on a low-memory or older laptop.

## Dataset

The dataset is not committed. Place the ASL Alphabet dataset at:

```text
archive/asl_alphabet_train/asl_alphabet_train/
  A/
  B/
  ...
  Z/
  del/
  nothing/
  space/
```

## Windows setup

From PowerShell at the repository root:

```powershell
py -3.12 -m venv backend/.venv
backend/.venv/Scripts/python -m pip install -r backend/requirements.txt
Set-Location frontend
npm install
Set-Location ..
```

## Train the classifier

Start with a smaller extraction to verify the pipeline:

```powershell
backend/.venv/Scripts/python backend/scripts/extract_landmarks.py --max-per-class 500
backend/.venv/Scripts/python backend/scripts/train_model.py
```

For the full available dataset:

```powershell
backend/.venv/Scripts/python backend/scripts/extract_landmarks.py --max-per-class 0
backend/.venv/Scripts/python backend/scripts/train_model.py
```

Outputs are intentionally ignored by Git:

- `data/processed/landmarks.csv`
- `data/metrics/metrics.json`
- `models/signcoach_model.joblib`

If the backend is already running after training, restart it or call `POST /api/model/reload`.

## Run the app

Terminal 1:

```powershell
backend/.venv/Scripts/python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Terminal 2:

```powershell
Set-Location frontend
npm run dev
```

Open `http://127.0.0.1:5173`, allow camera access, and choose Practice, Quiz, Freestyle, or Dashboard.
The frontend and backend bind only to `127.0.0.1`.

## Validation

```powershell
backend/.venv/Scripts/python -m compileall backend/app backend/scripts
backend/.venv/Scripts/python -m unittest discover -s backend/tests
Set-Location frontend
npm run build
```

Webcam frames are sent only to the local backend and are not saved by default.
