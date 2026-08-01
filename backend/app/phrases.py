from __future__ import annotations

from dataclasses import dataclass

import numpy as np


PHRASES = [
    {
        "id": "yes",
        "label": "Yes",
        "difficulty": "Starter",
        "durationSeconds": 1.6,
        "steps": [
            "Make a fist at about shoulder height.",
            "Keep palm facing sideways.",
            "Nod the fist down and up twice, like a head nod.",
        ],
        "checks": ["vertical_motion", "compact_hand"],
    },
    {
        "id": "no",
        "label": "No",
        "difficulty": "Starter",
        "durationSeconds": 1.6,
        "steps": [
            "Hold index and middle fingers extended with thumb open.",
            "Tap index and middle fingers down to meet the thumb.",
            "Repeat the closing motion once or twice.",
        ],
        "checks": ["pinch_motion"],
    },
    {
        "id": "good",
        "label": "Good",
        "difficulty": "Guided",
        "durationSeconds": 2.0,
        "steps": [
            "Start with flat fingers near the mouth or chin.",
            "Move the hand outward and down toward the other palm or space in front of you.",
            "Keep the movement smooth and direct.",
        ],
        "checks": ["outward_motion"],
    },
    {
        "id": "morning",
        "label": "Morning",
        "difficulty": "Guided",
        "durationSeconds": 2.2,
        "steps": [
            "Use one forearm as the horizon.",
            "Bring the other hand upward like the sun rising.",
            "This is normally a two-hand sign, so this MVP checks only rough rising motion.",
        ],
        "checks": ["rising_motion"],
    },
    {
        "id": "how-are-you",
        "label": "How are you?",
        "difficulty": "Guided",
        "durationSeconds": 2.5,
        "steps": [
            "Start with both hands curved inward near the chest.",
            "Rotate hands outward for HOW.",
            "Point gently toward the other person for YOU.",
            "This MVP gives coarse motion feedback; full scoring needs two-hand pose modeling.",
        ],
        "checks": ["movement_present"],
    },
    {
        "id": "i-am-good",
        "label": "I am good",
        "difficulty": "Guided",
        "durationSeconds": 2.5,
        "steps": [
            "Point to yourself for I.",
            "Then sign GOOD: flat hand from mouth/chin outward.",
            "This MVP checks that a smooth outward motion is present.",
        ],
        "checks": ["outward_motion"],
    },
]


@dataclass(frozen=True)
class PhraseScore:
    score: int
    passed: bool
    feedback: str
    metrics: dict[str, float]


def phrase_catalog() -> list[dict[str, object]]:
    return PHRASES


def analyze_phrase(phrase_id: str, frames: list[list[dict[str, float]]]) -> PhraseScore:
    phrase = next((item for item in PHRASES if item["id"] == phrase_id), None)
    if phrase is None:
        return PhraseScore(0, False, "Unknown phrase.", {})
    if len(frames) < 8:
        return PhraseScore(0, False, "Record a longer attempt with your hand visible.", {"frames": float(len(frames))})

    wrist = trajectory(frames, 0)
    index_tip = trajectory(frames, 8)
    middle_tip = trajectory(frames, 12)
    thumb_tip = trajectory(frames, 4)
    if wrist.size == 0:
        return PhraseScore(0, False, "No stable hand sequence found.", {"frames": float(len(frames))})

    metrics = movement_metrics(wrist)
    if phrase_id == "yes":
        return score_yes(metrics, frames)
    if phrase_id == "no":
        return score_no(index_tip, middle_tip, thumb_tip, metrics)
    if "outward_motion" in phrase["checks"]:
        return score_directional(metrics, "Move the hand farther outward from the body.", prefer_x=True)
    if "rising_motion" in phrase["checks"]:
        return score_directional(metrics, "Make the rising motion larger and clearer.", prefer_y=True)
    return score_general_motion(metrics)


def trajectory(frames: list[list[dict[str, float]]], index: int) -> np.ndarray:
    points = []
    for landmarks in frames:
        if len(landmarks) == 21:
            points.append([landmarks[index]["x"], landmarks[index]["y"], landmarks[index]["z"]])
    return np.asarray(points, dtype=np.float32)


def movement_metrics(points: np.ndarray) -> dict[str, float]:
    deltas = np.diff(points[:, :2], axis=0)
    path = float(np.sum(np.linalg.norm(deltas, axis=1))) if len(deltas) else 0.0
    displacement = points[-1, :2] - points[0, :2]
    x_range = float(np.ptp(points[:, 0]))
    y_range = float(np.ptp(points[:, 1]))
    direction_changes_y = float(np.sum(np.diff(np.sign(np.diff(points[:, 1]))) != 0)) if len(points) > 2 else 0.0
    return {
        "frames": float(len(points)),
        "path": path,
        "xRange": x_range,
        "yRange": y_range,
        "xDisplacement": float(displacement[0]),
        "yDisplacement": float(displacement[1]),
        "yDirectionChanges": direction_changes_y,
    }


def score_yes(metrics: dict[str, float], frames: list[list[dict[str, float]]]) -> PhraseScore:
    compact = average_finger_spread(frames) < 0.33
    vertical = metrics["yRange"] > 0.045 and metrics["yDirectionChanges"] >= 1
    score = int((45 if compact else 15) + min(metrics["yRange"] / 0.08, 1.0) * 35 + min(metrics["yDirectionChanges"] / 2, 1.0) * 20)
    if compact and vertical:
        return PhraseScore(min(score, 100), True, "Good YES motion. The fist nod was visible.", metrics)
    if not compact:
        return PhraseScore(score, False, "Make a tighter fist for YES, then nod it down and up.", metrics)
    return PhraseScore(score, False, "Make the YES nod larger: move the fist down and back up.", metrics)


def score_no(index_tip: np.ndarray, middle_tip: np.ndarray, thumb_tip: np.ndarray, metrics: dict[str, float]) -> PhraseScore:
    if len(index_tip) < 8:
        return PhraseScore(0, False, "Keep your hand visible for the full NO motion.", metrics)
    pinch_distance = (np.linalg.norm(index_tip[:, :2] - thumb_tip[:, :2], axis=1) + np.linalg.norm(middle_tip[:, :2] - thumb_tip[:, :2], axis=1)) / 2
    distance_change = float(np.ptp(pinch_distance))
    closes = float(np.min(pinch_distance))
    score = int(min(distance_change / 0.08, 1.0) * 65 + (35 if closes < 0.16 else 10))
    if distance_change > 0.045 and closes < 0.18:
        return PhraseScore(min(score, 100), True, "Good NO motion. The fingers closed toward the thumb.", {**metrics, "pinchChange": distance_change})
    return PhraseScore(score, False, "For NO, close index and middle fingers down to the thumb more clearly.", {**metrics, "pinchChange": distance_change})


def score_directional(metrics: dict[str, float], fallback: str, *, prefer_x: bool = False, prefer_y: bool = False) -> PhraseScore:
    axis_motion = abs(metrics["xDisplacement"]) if prefer_x else abs(metrics["yDisplacement"]) if prefer_y else metrics["path"]
    score = int(min(axis_motion / 0.16, 1.0) * 70 + min(metrics["path"] / 0.25, 1.0) * 30)
    if score >= 55:
        return PhraseScore(score, True, "Good motion for a guided phrase attempt.", metrics)
    return PhraseScore(score, False, fallback, metrics)


def score_general_motion(metrics: dict[str, float]) -> PhraseScore:
    score = int(min(metrics["path"] / 0.22, 1.0) * 100)
    if score >= 55:
        return PhraseScore(score, True, "Good practice motion. Full phrase recognition will need a trained video model.", metrics)
    return PhraseScore(score, False, "Make the phrase movement larger and keep your hand visible.", metrics)


def average_finger_spread(frames: list[list[dict[str, float]]]) -> float:
    spreads = []
    for landmarks in frames:
        if len(landmarks) != 21:
            continue
        tips = np.asarray([[landmarks[index]["x"], landmarks[index]["y"]] for index in (4, 8, 12, 16, 20)], dtype=np.float32)
        wrist = np.asarray([landmarks[0]["x"], landmarks[0]["y"]], dtype=np.float32)
        scale = max(float(np.max(np.linalg.norm(tips - wrist, axis=1))), 1e-6)
        spreads.append(float(np.mean(np.linalg.norm(tips - tips.mean(axis=0), axis=1))) / scale)
    return float(np.mean(spreads)) if spreads else 1.0
