from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from app.config import MAX_FRAME_HEIGHT, MAX_FRAME_WIDTH

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover - exercised only before dependencies are installed
    mp = None


@dataclass
class LandmarkResult:
    hand_detected: bool
    landmarks: list[dict[str, float]]
    connections: list[tuple[int, int]]
    handedness: str | None
    bounding_box: dict[str, float] | None
    framing_score: float
    message: str


class HandLandmarkDetector:
    def __init__(self, static_image_mode: bool = False) -> None:
        self._hands = None
        self._connections: list[tuple[int, int]] = []

        if mp is not None:
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=static_image_mode,
                max_num_hands=1,
                model_complexity=1,
                min_detection_confidence=0.55,
                min_tracking_confidence=0.55,
            )
            self._connections = [(start, end) for start, end in mp.solutions.hands.HAND_CONNECTIONS]

    @property
    def available(self) -> bool:
        return self._hands is not None

    def detect(self, frame_bgr: np.ndarray) -> LandmarkResult:
        if self._hands is None:
            return LandmarkResult(
                hand_detected=False,
                landmarks=[],
                connections=[],
                handedness=None,
                bounding_box=None,
                framing_score=0.0,
                message="MediaPipe is not installed. Install backend dependencies first.",
            )

        frame_bgr = resize_frame(frame_bgr)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._hands.process(frame_rgb)

        if not results.multi_hand_landmarks:
            return LandmarkResult(
                hand_detected=False,
                landmarks=[],
                connections=self._connections,
                handedness=None,
                bounding_box=None,
                framing_score=0.0,
                message="Move your hand into frame.",
            )

        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = [
            {"x": lm.x, "y": lm.y, "z": lm.z}
            for lm in hand_landmarks.landmark
        ]
        handedness = None
        if results.multi_handedness:
            handedness = results.multi_handedness[0].classification[0].label

        bbox = bounding_box_for(landmarks)
        framing_score = score_framing(bbox)
        if framing_score < 0.45:
            message = "Move closer to the camera."
        else:
            message = "Hand detected."

        return LandmarkResult(
            hand_detected=True,
            landmarks=landmarks,
            connections=self._connections,
            handedness=handedness,
            bounding_box=bbox,
            framing_score=framing_score,
            message=message,
        )


def decode_data_url(data_url: str) -> np.ndarray:
    if "," in data_url:
        _, payload = data_url.split(",", 1)
    else:
        payload = data_url

    image_bytes = base64.b64decode(payload)
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Unable to decode image frame.")
    return frame


def resize_frame(frame_bgr: np.ndarray) -> np.ndarray:
    height, width = frame_bgr.shape[:2]
    scale = min(MAX_FRAME_WIDTH / width, MAX_FRAME_HEIGHT / height, 1.0)
    if scale >= 1.0:
        return frame_bgr
    return cv2.resize(frame_bgr, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)


def bounding_box_for(landmarks: list[dict[str, float]]) -> dict[str, float]:
    xs = [lm["x"] for lm in landmarks]
    ys = [lm["y"] for lm in landmarks]
    left = max(0.0, min(xs))
    top = max(0.0, min(ys))
    right = min(1.0, max(xs))
    bottom = min(1.0, max(ys))
    return {
        "x": left,
        "y": top,
        "width": max(0.0, right - left),
        "height": max(0.0, bottom - top),
    }


def score_framing(bbox: dict[str, float] | None) -> float:
    if bbox is None:
        return 0.0

    area = bbox["width"] * bbox["height"]
    center_x = bbox["x"] + bbox["width"] / 2
    center_y = bbox["y"] + bbox["height"] / 2
    center_distance = ((center_x - 0.5) ** 2 + (center_y - 0.5) ** 2) ** 0.5

    size_score = min(area / 0.12, 1.0)
    center_score = max(0.0, 1.0 - center_distance / 0.45)
    return round((size_score * 0.65) + (center_score * 0.35), 3)


def result_to_payload(result: LandmarkResult) -> dict[str, Any]:
    return {
        "handDetected": result.hand_detected,
        "landmarks": result.landmarks,
        "connections": result.connections,
        "handedness": result.handedness,
        "boundingBox": result.bounding_box,
        "framingScore": result.framing_score,
        "message": result.message,
        "phase": "landmarks",
    }

