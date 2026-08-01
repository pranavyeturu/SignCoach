from __future__ import annotations

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.calibration import append_calibration_sample, calibration_summary, train_with_calibration
from app.config import DATASET_DIR, MODEL_PATH, PHRASE_MODEL_PATH, SUPPORTED_IMAGE_EXTENSIONS
from app.disambiguation import a_t_shape_cue
from app.feedback import feedback_for
from app.landmarks import HandLandmarkDetector, decode_data_url, result_to_payload
from app.phrases import analyze_phrase, phrase_catalog
from app.phrase_prediction import PhraseSignPredictor
from app.prediction import SignPredictor
from app.scoring import attempt_score
from app.schemas import CalibrationSampleRequest, DatasetSummary, FrameRequest, PhraseAnalyzeRequest
from app.smoothing import PredictionSmoother

app = FastAPI(title="SignCoach AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = HandLandmarkDetector()
predictor = SignPredictor(MODEL_PATH)
phrase_predictor = PhraseSignPredictor(PHRASE_MODEL_PATH)


@app.get("/api/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "mediapipeAvailable": detector.available,
        "modelAvailable": predictor.available,
        "modelError": predictor.error,
        "phraseModelAvailable": phrase_predictor.available,
        "phraseModelError": phrase_predictor.error,
        "phase": "complete-mvp",
    }


@app.get("/api/dataset", response_model=DatasetSummary)
def dataset_summary() -> DatasetSummary:
    classes = []
    image_count = 0

    if DATASET_DIR.exists():
        for child in sorted(DATASET_DIR.iterdir()):
            if child.is_dir():
                classes.append(child.name)
                image_count += sum(
                    1
                    for item in child.iterdir()
                    if item.is_file() and item.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
                )

    return DatasetSummary(path=str(DATASET_DIR), classes=classes, image_count=image_count)


@app.get("/api/reference/{letter}")
def reference_image(letter: str) -> FileResponse:
    label = letter.upper()
    if len(label) != 1 or not label.isalpha():
        raise HTTPException(status_code=404, detail="Reference image not found.")

    class_dir = DATASET_DIR / label
    if not class_dir.exists() or not class_dir.is_dir():
        raise HTTPException(status_code=404, detail="Reference image not found.")

    for image in sorted(class_dir.iterdir()):
        if image.is_file() and image.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            return FileResponse(image)

    raise HTTPException(status_code=404, detail="Reference image not found.")


@app.post("/api/detect")
def detect_frame(request: FrameRequest) -> dict[str, object]:
    frame = decode_data_url(request.image)
    result = detector.detect(frame)
    return prediction_payload(result, PredictionSmoother())


@app.post("/api/model/reload")
def reload_model() -> dict[str, object]:
    predictor.reload()
    return {"modelAvailable": predictor.available, "modelError": predictor.error}


@app.get("/api/calibration")
def get_calibration() -> dict[str, object]:
    return calibration_summary()


@app.get("/api/phrases")
def get_phrases() -> list[dict[str, object]]:
    return phrase_catalog()


@app.post("/api/phrases/analyze")
def analyze_phrase_attempt(request: PhraseAnalyzeRequest) -> dict[str, object]:
    result = analyze_phrase(request.phrase_id, request.frames)
    model_prediction = phrase_predictor.predict(request.frames) if phrase_predictor.available else None
    target = phrase_id_to_label(request.phrase_id)
    model_passed = bool(model_prediction and target and model_prediction.label == target and model_prediction.confidence >= 0.45)
    return {
        "score": result.score,
        "passed": result.passed or model_passed,
        "feedback": model_feedback(model_prediction, target) if model_prediction else result.feedback,
        "metrics": result.metrics,
        "modelAvailable": phrase_predictor.available,
        "modelError": phrase_predictor.error,
        "modelPrediction": model_prediction.label if model_prediction else None,
        "modelConfidence": model_prediction.confidence if model_prediction else 0.0,
        "topPredictions": model_prediction.top_predictions if model_prediction else [],
        "ruleFeedback": result.feedback,
    }


@app.post("/api/phrases/model/reload")
def reload_phrase_model() -> dict[str, object]:
    phrase_predictor.reload()
    return {"phraseModelAvailable": phrase_predictor.available, "phraseModelError": phrase_predictor.error}


@app.post("/api/calibration/sample")
def save_calibration_sample(request: CalibrationSampleRequest) -> dict[str, object]:
    try:
        return append_calibration_sample(request.label, request.landmarks)
    except ValueError as exc:
        return {"error": str(exc), **calibration_summary()}


@app.post("/api/calibration/train")
def train_calibrated_model() -> dict[str, object]:
    try:
        metrics = train_with_calibration()
        predictor.reload()
        return {
            "modelAvailable": predictor.available,
            "modelError": predictor.error,
            "metrics": metrics,
            "calibration": calibration_summary(),
        }
    except Exception as exc:
        return {
            "modelAvailable": predictor.available,
            "modelError": str(exc),
            "calibration": calibration_summary(),
        }


@app.websocket("/ws/detect")
async def detect_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    smoother = PredictionSmoother()
    try:
        while True:
            data = await websocket.receive_json()
            frame = decode_data_url(data["image"])
            result = detector.detect(frame)
            await websocket.send_json(
                prediction_payload(result, smoother, str(data.get("target") or "").upper() or None)
            )
    except WebSocketDisconnect:
        return
    except Exception as exc:
        await websocket.send_json(
            {
                "handDetected": False,
                "landmarks": [],
                "connections": [],
                "handedness": None,
                "rawHandedness": None,
                "boundingBox": None,
                "framingScore": 0,
                "message": str(exc),
                "phase": "error",
            }
        )


def prediction_payload(result, smoother: PredictionSmoother, target: str | None = None) -> dict[str, object]:
    payload = result_to_payload(result)
    raw = predictor.predict(result.landmarks) if result.hand_detected else None
    shape_cue = a_t_shape_cue(result.landmarks) if result.hand_detected else None
    adjusted_label = raw.label if raw else None
    adjusted_confidence = raw.confidence if raw else 0.0
    adjustment_reason = None
    if raw and target and target.upper() in {"A", "T"} and raw.label in {"A", "T"} and shape_cue and shape_cue.label:
        if shape_cue.label == target.upper() and shape_cue.label != raw.label:
            adjusted_label = shape_cue.label
            adjusted_confidence = max(raw.confidence, shape_cue.confidence)
            adjustment_reason = shape_cue.reason

    if raw is None:
        smoother.reset()
        smooth = {"label": None, "confidence": 0.0, "stable": False, "holdProgress": 0.0}
    else:
        smooth = smoother.add(
            adjusted_label,
            adjusted_confidence,
            threshold=confidence_threshold_for(target, adjusted_label),
        )

    label = smooth["label"]
    confidence = float(smooth["confidence"])
    stable = bool(smooth["stable"])
    payload.update(
        {
            "prediction": label,
            "rawPrediction": raw.label if raw else None,
            "adjustedPrediction": adjusted_label,
            "adjustmentReason": adjustment_reason,
            "shapeCue": shape_cue.label if shape_cue else None,
            "confidence": confidence,
            "topPredictions": raw.top_predictions if raw else [],
            "stable": stable,
            "holdProgress": smooth["holdProgress"],
            "confirmed": bool(target and label == target and stable),
            "target": target,
            "attemptScore": attempt_score(confidence, float(smooth["holdProgress"]), result.framing_score),
            "modelAvailable": predictor.available,
            "message": feedback_for(
                hand_detected=result.hand_detected,
                framing_score=result.framing_score,
                predicted=str(label) if label else None,
                confidence=confidence,
                target=target,
                stable=stable,
            ) if predictor.available else (predictor.error or result.message),
            "phase": "prediction",
        }
    )
    return payload


def confidence_threshold_for(target: str | None, predicted: str | None) -> float:
    if target and predicted and target.upper() == predicted.upper() and target.upper() in {"A", "R", "T", "U"}:
        return 0.65
    return 0.55


def phrase_id_to_label(phrase_id: str) -> str | None:
    mapping = {
        "yes": "yes",
        "no": "no",
        "good": "good",
        "morning": "morning",
        "how-are-you": "how",
        "i-am-good": "good",
    }
    return mapping.get(phrase_id)


def model_feedback(prediction, target: str | None) -> str:
    if prediction is None:
        return "No phrase model prediction available."
    confidence = round(prediction.confidence * 100)
    if target and prediction.label == target:
        return f"MS-ASL model detected {prediction.label.upper()} with {confidence}% confidence."
    if target:
        return f"MS-ASL model detected {prediction.label.upper()}, expected {target.upper()}."
    return f"MS-ASL model detected {prediction.label.upper()} with {confidence}% confidence."
