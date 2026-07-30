from __future__ import annotations

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.config import DATASET_DIR, SUPPORTED_IMAGE_EXTENSIONS
from app.landmarks import HandLandmarkDetector, decode_data_url, result_to_payload
from app.schemas import DatasetSummary, FrameRequest

app = FastAPI(title="SignCoach AI API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = HandLandmarkDetector()


@app.get("/api/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "mediapipeAvailable": detector.available,
        "phase": "phase-1-landmarks",
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


@app.post("/api/detect")
def detect_frame(request: FrameRequest) -> dict[str, object]:
    frame = decode_data_url(request.image)
    result = detector.detect(frame)
    return result_to_payload(result)


@app.websocket("/ws/detect")
async def detect_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            frame = decode_data_url(data["image"])
            result = detector.detect(frame)
            await websocket.send_json(result_to_payload(result))
    except WebSocketDisconnect:
        return
    except Exception as exc:
        await websocket.send_json(
            {
                "handDetected": False,
                "landmarks": [],
                "connections": [],
                "handedness": None,
                "boundingBox": None,
                "framingScore": 0,
                "message": str(exc),
                "phase": "error",
            }
        )

