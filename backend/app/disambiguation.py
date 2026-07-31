from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ShapeCue:
    label: str | None
    confidence: float
    reason: str | None


def a_t_shape_cue(landmarks: list[dict[str, float]]) -> ShapeCue:
    points = normalized_points(landmarks)
    thumb_tip = points[4]
    index_mcp = points[5]
    middle_mcp = points[9]
    index_tip = points[8]
    middle_tip = points[12]

    thumb_index_mcp = distance(thumb_tip, index_mcp)
    thumb_middle_mcp = distance(thumb_tip, middle_mcp)
    thumb_index_tip = distance(thumb_tip, index_tip)
    thumb_middle_tip = distance(thumb_tip, middle_tip)

    t_votes = [
        thumb_index_mcp > 0.38,
        thumb_middle_mcp > 0.54,
        thumb_middle_tip > 0.60,
        thumb_index_tip > 0.42,
    ]
    a_votes = [
        thumb_index_mcp < 0.37,
        thumb_middle_mcp < 0.52,
        thumb_middle_tip < 0.60,
        thumb_index_tip < 0.43,
    ]

    t_score = sum(t_votes)
    a_score = sum(a_votes)
    if t_score >= 3 and t_score > a_score:
        return ShapeCue("T", min(0.78, 0.62 + (t_score * 0.04)), "thumb geometry matches T")
    if a_score >= 3 and a_score > t_score:
        return ShapeCue("A", min(0.78, 0.62 + (a_score * 0.04)), "thumb geometry matches A")
    return ShapeCue(None, 0.0, None)


def normalized_points(landmarks: list[dict[str, float]]) -> np.ndarray:
    if len(landmarks) != 21:
        raise ValueError("Expected exactly 21 hand landmarks.")
    points = np.asarray([[point["x"], point["y"], point["z"]] for point in landmarks], dtype=np.float32)
    points -= points[0]
    scale = float(np.max(np.linalg.norm(points[:, :2], axis=1)))
    if scale < 1e-6:
        raise ValueError("Landmarks are too close together to normalize.")
    return points / scale


def distance(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.linalg.norm(first - second))

