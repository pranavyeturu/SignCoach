# PRD: SignCoach AI

## Realtime ASL Alphabet Teacher

## 1. Overview

SignCoach AI is a realtime computer vision app that helps users learn ASL fingerspelling. The app uses a webcam to detect hand landmarks, classify ASL alphabet signs, compare the user's sign against a target letter, and provide instant scoring and feedback.

The MVP focuses on static ASL alphabet signs, not full ASL sentence translation.

## 2. Product Goal

Build a realtime AI/ML sign language learning prototype by **August 2, 2026, 12:00 AM**.

Primary goal:

> Help users practice ASL alphabet signs with realtime visual feedback.

## 3. Target Users

- Beginners learning ASL fingerspelling
- Students building a computer vision project
- Accessibility-tech enthusiasts
- Teachers or demo audiences evaluating realtime AI/ML applications

## 4. Problem

Most sign language recognition projects only classify signs. They do not help users improve.

A learner needs:

- Realtime correction
- Confidence feedback
- Practice flow
- Progress tracking
- Clear indication of what the model sees

## 5. Solution

SignCoach AI acts as a realtime ASL tutor.

The user selects or is assigned a target letter. The webcam detects their hand pose, predicts the signed letter, scores the attempt, and gives feedback such as:

- "Correct, hold steady."
- "Detected B, target is A."
- "Move hand closer to camera."
- "Confidence is low. Face palm toward camera."
- "No hand detected."

## 6. MVP Scope

### In Scope

- Webcam input
- Realtime hand landmark detection
- ASL alphabet classification
- Target letter practice mode
- Freestyle prediction mode
- Confidence score
- Basic feedback messages
- Session score
- Per-letter progress tracking
- Simple dashboard
- Model training notebook or script
- README and demo instructions

### Out of Scope

- Full ASL sentence translation
- Word-level ASL recognition
- Dynamic fluent sign phrases
- Production authentication
- Cloud sync
- Mobile app
- Medical or official accessibility certification
- Perfect accuracy across all lighting and camera conditions

## 7. Core Features

### Feature 1: Webcam Practice

User can start webcam and see a live video feed.

Requirements:

- Display webcam stream
- Detect hand in realtime
- Draw hand landmarks
- Show status:
  - Hand detected
  - No hand detected
  - Low confidence
  - Prediction ready

### Feature 2: ASL Letter Recognition

App predicts the ASL alphabet letter from webcam input.

Requirements:

- Support A-Z static letters where feasible
- Handle `space`, `delete`, or `nothing` only if dataset/model supports it
- Show top prediction
- Show confidence percentage
- Apply confidence threshold before marking a sign as correct

### Feature 3: Teacher Mode

User practices a selected target letter.

Requirements:

- User selects target letter
- App compares prediction to target
- App shows correct/incorrect state
- User must hold correct sign briefly to count success
- App shows current attempt score

### Feature 4: Quiz Mode

App randomly prompts user with letters.

Requirements:

- Random target letter
- Timer or hold-to-confirm mechanic
- Track correct attempts
- Track incorrect attempts
- Move to next letter after success

### Feature 5: Feedback Engine

App gives simple actionable feedback.

Initial rules:

- No hand detected: "Move your hand into frame."
- Hand too small: "Move closer to the camera."
- Low confidence: "Hold steady and face your palm toward the camera."
- Wrong letter: "Detected X, target is Y."
- Correct letter: "Correct. Hold for one second."
- Unstable hand: "Hold your hand steady."

### Feature 6: Progress Dashboard

App tracks learning progress during a session.

Requirements:

- Overall accuracy
- Attempts count
- Correct count
- Current streak
- Best streak
- Accuracy by letter
- Hardest letters based on mistakes
- Session duration

## 8. User Flows

### Flow 1: Practice a Letter

1. User opens app.
2. User clicks "Start Camera."
3. User chooses target letter.
4. User makes the sign.
5. App detects landmarks.
6. App predicts letter.
7. App compares prediction to target.
8. App provides feedback.
9. User holds correct sign.
10. App records success.

### Flow 2: Quiz Mode

1. User starts quiz.
2. App shows random letter.
3. User signs the letter.
4. App predicts result.
5. If correct and held steady, app increments score.
6. App moves to next letter.
7. End screen shows accuracy and hard letters.

### Flow 3: Freestyle Mode

1. User starts camera.
2. User signs any supported letter.
3. App shows prediction and confidence.
4. App updates live as hand pose changes.

## 9. Functional Requirements

- The app must access webcam input.
- The app must run hand tracking in realtime.
- The app must classify signs from hand landmarks or image frames.
- The app must show prediction confidence.
- The app must compare current prediction against target letter.
- The app must show feedback in under 500 ms after prediction.
- The app must track session-level metrics.
- The app must save progress locally for the session.
- The app must run locally on a laptop.

## 10. Non-Functional Requirements

- Realtime performance target: at least 10 FPS.
- Prediction latency target: under 500 ms.
- App should run without login.
- App should not upload webcam frames unless explicitly documented.
- UI should be simple and demo-ready.
- Model should be small enough to run locally.
- System should fail gracefully when hand is not visible.

## 11. Dataset

### Recommended MVP Dataset

**ASL Alphabet Dataset**

- Static ASL alphabet images
- Classes: A-Z plus optional `space`, `delete`, and `nothing`, depending on dataset
- Good for first model

### Alternative Simpler Dataset

**Sign Language MNIST**

- 24 static ASL letters
- Excludes J and Z because they require motion
- Easier to train quickly

### Future Dataset

**WLASL**

- Word-level ASL video dataset
- Better for future dynamic signs
- Too large for MVP timeline

## 12. ML Approach

Recommended approach:

1. Use MediaPipe Hands to detect 21 hand landmarks.
2. Convert landmarks into normalized feature vectors.
3. Train a classifier on extracted landmarks.
4. Use classifier for realtime predictions.

Feature vector:

- 21 hand landmarks
- X, Y, Z coordinates
- Normalized relative to wrist
- Optional finger distances and angles

Model options:

- Random Forest for fastest MVP
- SVM for baseline
- MLP for improved accuracy
- CNN if using raw image classification

MVP recommendation:

> MediaPipe Hands + Random Forest or SVM

## 13. Model Training Pipeline

Steps:

1. Download dataset.
2. Load each image.
3. Run MediaPipe Hands.
4. Extract landmarks.
5. Normalize landmarks.
6. Save features and labels.
7. Train classifier.
8. Evaluate accuracy.
9. Save model.
10. Load model in realtime app.

Artifacts:

- `train_model.py`
- `model.pkl`
- `label_encoder.pkl`
- `metrics.json`
- `confusion_matrix.png`

## 14. Realtime Prediction Pipeline

Steps per frame:

1. Capture webcam frame.
2. Run MediaPipe hand detection.
3. If no hand is detected, show "No hand detected."
4. Extract landmarks.
5. Normalize features.
6. Run classifier.
7. Get prediction and confidence.
8. Apply smoothing over recent frames.
9. Compare against target.
10. Update feedback and metrics.

## 15. Prediction Smoothing

To reduce flicker:

- Keep the last 10 predictions.
- Use majority vote.
- Require confidence above threshold.
- Require the same prediction for 0.5 to 1.0 seconds before confirming success.

Suggested defaults:

- Confidence threshold: 70%
- Hold duration: 1 second
- Prediction window: 10 frames

## 16. Scoring

Attempt score:

```text
score = confidence_score * 0.70
      + stability_score * 0.15
      + framing_score * 0.15
```

Session metrics:

- Total attempts
- Correct attempts
- Incorrect attempts
- Accuracy
- Streak
- Per-letter accuracy
- Hardest letters
- Average confidence

## 17. UI Requirements

Main layout:

- Left: webcam feed with landmarks
- Right: target letter, prediction, confidence, feedback
- Bottom: session stats

Pages or tabs:

- Practice
- Quiz
- Freestyle
- Dashboard

Practice screen:

- Letter selector
- Target sign reference image
- Live prediction
- Feedback message
- Score

Dashboard:

- Accuracy
- Attempts
- Streak
- Hardest letters
- Per-letter progress

## 18. Tech Stack

Recommended:

- Python
- OpenCV
- MediaPipe
- scikit-learn
- Streamlit
- NumPy
- Pandas
- Matplotlib or Plotly

Optional:

- TensorFlow or PyTorch for neural model
- FastAPI if separating backend and frontend

Fastest MVP:

> Single Streamlit app

## 19. Success Metrics

### Technical

- Webcam runs reliably
- Hand landmarks are detected in realtime
- Classifier predicts signs with usable accuracy
- App responds within 500 ms
- Demo works for at least 8-10 letters reliably

### Product

- User can practice a target letter
- User receives immediate feedback
- User can see progress
- User understands mistakes
- Demo audience understands value in under 30 seconds

## 20. Risks

### Risk: Dataset Images May Not Match Webcam Conditions

Mitigation:

- Use MediaPipe landmarks instead of raw images.
- Normalize landmarks.
- Test live and adjust.
- Focus demo on reliable letters.

### Risk: Some Letters Are Visually Similar

Examples:

- M/N/T
- A/S
- E/O

Mitigation:

- Add confidence threshold.
- Show "uncertain" state.
- Track top 3 predictions.

### Risk: J and Z Require Motion

Mitigation:

- Exclude them from MVP or mark them as future dynamic signs.

### Risk: Realtime Predictions Flicker

Mitigation:

- Add rolling majority vote and hold-to-confirm.

### Risk: Feedback May Be Too Generic

Mitigation:

- Start with practical camera and pose feedback.
- Add letter-specific feedback later.

## 21. Milestones

### Milestone 1: Webcam + Landmarks

- Webcam feed working
- MediaPipe landmarks visible
- No-hand detection working

### Milestone 2: Dataset Training

- Dataset loaded
- Landmarks extracted
- Classifier trained
- Model saved

### Milestone 3: Realtime Recognition

- Webcam frames classified
- Prediction and confidence shown
- Smoothing added

### Milestone 4: Teacher Mode

- Target letter selection
- Correct/incorrect comparison
- Hold-to-confirm
- Feedback messages

### Milestone 5: Dashboard

- Attempts, accuracy, streak
- Per-letter performance
- End-session summary

## 22. Timeline To August 2

### July 31

- Build Streamlit UI
- Add webcam
- Add MediaPipe hand landmarks
- Add no-hand and low-confidence states

### August 1 Morning

- Train model from dataset
- Save classifier
- Add realtime prediction

### August 1 Afternoon

- Add teacher mode
- Add quiz mode
- Add smoothing and hold-to-confirm

### August 1 Night

- Add dashboard
- Add README
- Test demo letters
- Prepare final presentation/demo

## 23. Future Scope

- Dynamic signs like J and Z
- Word-level sign recognition using WLASL
- Sentence-level translation
- Personalized model calibration
- Two-hand signs
- Teacher-created lesson plans
- Voice pronunciation of letters
- Multiplayer classroom mode
- Accessibility integrations
- Mobile version

## 24. Demo Script

1. Open SignCoach AI.
2. Click "Start Camera."
3. Select target letter `A`.
4. Make the wrong sign first.
5. App says "Detected B, target is A."
6. Make the correct sign.
7. App says "Correct. Hold for one second."
8. Score updates.
9. Start quiz mode.
10. Complete 5 random letters.
11. Show dashboard with accuracy and hardest letters.

## 25. One-Line Pitch

SignCoach AI is a realtime computer vision tutor that helps users learn ASL fingerspelling by detecting hand signs, giving instant feedback, and tracking progress letter by letter.
