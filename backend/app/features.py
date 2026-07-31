from __future__ import annotations

import numpy as np


def normalize_landmarks(landmarks: list[dict[str, float]]) -> np.ndarray:
    """Return a translation/scale-normalized 63-value landmark vector."""
    if len(landmarks) != 21:
        raise ValueError("Expected exactly 21 hand landmarks.")

    points = np.asarray(
        [[point["x"], point["y"], point["z"]] for point in landmarks],
        dtype=np.float32,
    )
    points -= points[0]
    scale = float(np.max(np.linalg.norm(points[:, :2], axis=1)))
    if scale < 1e-6:
        raise ValueError("Landmarks are too close together to normalize.")
    return (points / scale).reshape(-1)

