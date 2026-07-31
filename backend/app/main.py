from __future__ import annotations

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.calibration import append_calibration_sample, calibration_summary, train_with_calibration
from app.config import DATASET_DIR, MODEL_PATH, SUPPORTED_IMAGE_EXTENSIONS
from app.disambiguation import a_t_shape_cue
from app.feedback import feedback_for
from app.landmarks import HandLandmarkDetector, decode_data_url, result_to_payload
from app.prediction import SignPredictor
from app.scoring import attempt_score
from app.schemas import CalibrationSampleRequest, DatasetSummary, FrameRequest
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


@app.get("/api/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "mediapipeAvailable": detector.available,
        "modelAvailable": predictor.available,
        "modelError": predictor.error,
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
