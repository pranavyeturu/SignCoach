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
- Phrase lesson mode for starter ASL signs such as `YES`, `NO`, `GOOD`, and `MORNING`
- Rule-based motion scoring for short phrase attempts, strongest for `YES` and `NO`
- Short calibration mode for collecting live webcam samples of difficult letters
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

## Calibrate for a user

Use Calibration mode when a few letters are failing under the user's real webcam, lighting, or hand
shape.

This is especially useful for visually similar pairs such as `U/R` and `A/T`. The classifier uses
extra geometric features for these pairs, including fingertip distances, joint angles, finger
extension ratios, index/middle crossing cues, and thumb-position cues. Calibration adds live examples
from the user's actual camera setup.

1. Start the backend and frontend.
2. Open `http://127.0.0.1:5173`.
3. Start the camera.
4. Choose `Calibration`.
5. Select a difficult letter.
6. Hold the sign in normal practice position.
7. Click `Capture 12`.
8. Repeat for the letters that are failing.
9. Click `Retrain model`.

Calibration samples are saved locally to:

```text
data/calibration/landmarks.csv
```

Retraining combines the original extracted dataset landmarks with the user's calibration samples and
reloads the model in the running backend.

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

## ASL video dataset recommendation

For proper phrase/sign recognition, use an isolated-sign video dataset rather than the alphabet image
dataset.

Recommended first choice:

- WLASL: word-level ASL video dataset with 2,000 ASL lexical signs.
  - Official page: https://dxli94.github.io/WLASL/
  - Useful for training isolated word/sign recognition before attempting full sentence translation.

Other strong options:

- MS-ASL: large real-world ASL video dataset with over 25,000 annotated clips and 1,000 signs.
  - Microsoft download: https://www.microsoft.com/en-us/download/details.aspx?id=100121
- ASL Citizen: crowdsourced isolated ASL dataset with about 84,000 videos and 2,700+ signs.
  - Microsoft Research page: https://www.microsoft.com/en-us/research/project/asl-citizen/
- ASLLVD: linguistically rich lexicon video dataset with thousands of citation-form signs and detailed annotations.
  - Boston University page: https://www.bu.edu/asllrp/av/dai-asllvd.html

Phrase mode in this app is currently a lesson and rule-checking MVP. A trained phrase model should
use landmark/video sequences over time, not single-frame alphabet landmarks.

## Phase 3: MS-ASL phrase model

The `MS-ASL/` folder contains annotation JSON files. Those files reference online videos and clip
timestamps; they do not include the video files themselves.

Build a small starter-sign manifest:

```bash
cd backend
source .venv/bin/activate
python scripts/prepare_msasl_manifest.py --max-per-gloss 80
```

This creates:

```text
data/msasl/phrase_manifest.json
```

Download the referenced clips with `yt-dlp`:

```bash
python scripts/download_msasl_clips.py --limit 100
```

Remove `--limit 100` to download all manifest clips. Some source videos may be unavailable because
MS-ASL references public web videos that can disappear over time.
The first local smoke test downloaded 2 of 5 attempted clips; unavailable YouTube links are expected.

Extract sequence features and train the phrase model:

```bash
python scripts/extract_msasl_sequence_features.py --max-per-gloss 60
python scripts/train_phrase_model.py
curl -X POST http://127.0.0.1:8000/api/phrases/model/reload
```

Train only after enough clips have downloaded and extracted. As a practical minimum, aim for at
least 20 usable clips per starter sign.

Generated outputs are ignored by Git:

- `data/msasl/phrase_manifest.json`
- `data/msasl/videos/`
- `data/processed/msasl_phrase_features.csv`
- `models/phrase_sign_model.joblib`

When `models/phrase_sign_model.joblib` exists, Phrase mode combines the trained MS-ASL model with
the existing rule-based motion checks.
