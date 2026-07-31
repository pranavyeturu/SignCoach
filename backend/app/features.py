from __future__ import annotations

import numpy as np


TIP_LANDMARKS = [4, 8, 12, 16, 20]
FINGER_CHAINS = [
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
    (17, 18, 19, 20),
]
ANGLE_TRIPLES = [
    (1, 2, 3), (2, 3, 4),
    (5, 6, 7), (6, 7, 8),
    (9, 10, 11), (10, 11, 12),
    (13, 14, 15), (14, 15, 16),
    (17, 18, 19), (18, 19, 20),
    (0, 5, 9), (0, 9, 13), (0, 13, 17),
]
DISTANCE_PAIRS = [
    (4, 8), (4, 12), (4, 16), (4, 20),
    (8, 12), (8, 16), (8, 20),
    (12, 16), (12, 20),
    (16, 20),
    (5, 9), (9, 13), (13, 17),
    (8, 6), (12, 10), (16, 14), (20, 18),
]


def normalize_landmarks(landmarks: list[dict[str, float]]) -> np.ndarray:
    """Return normalized landmark coordinates plus hand-shape geometry features."""
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
    normalized = points / scale
    return np.concatenate(
        [
            normalized.reshape(-1),
            distance_features(normalized),
            angle_features(normalized),
            finger_extension_features(normalized),
            disambiguation_features(normalized),
        ]
    ).astype(np.float32)


def feature_names() -> list[str]:
    return [
        *[f"{axis}{index}" for index in range(21) for axis in ("x", "y", "z")],
        *[f"distance_{start}_{end}" for start, end in DISTANCE_PAIRS],
        *[f"angle_{start}_{middle}_{end}" for start, middle, end in ANGLE_TRIPLES],
        *[f"extension_{chain[-1]}" for chain in FINGER_CHAINS],
        "index_middle_tip_x_gap",
        "index_middle_tip_y_gap",
        "index_over_middle_cross",
        "thumb_between_index_middle_x",
        "thumb_tip_to_index_mcp",
        "thumb_tip_to_middle_mcp",
        "thumb_tip_to_index_tip",
        "thumb_tip_to_middle_tip",
    ]


def distance_features(points: np.ndarray) -> np.ndarray:
    return np.asarray([distance(points[start], points[end]) for start, end in DISTANCE_PAIRS], dtype=np.float32)


def angle_features(points: np.ndarray) -> np.ndarray:
    return np.asarray([joint_angle(points[start], points[middle], points[end]) for start, middle, end in ANGLE_TRIPLES], dtype=np.float32)


def finger_extension_features(points: np.ndarray) -> np.ndarray:
    values = []
    for chain in FINGER_CHAINS:
        base = points[chain[0]]
        tip = points[chain[-1]]
        path_length = sum(distance(points[chain[index]], points[chain[index + 1]]) for index in range(len(chain) - 1))
        values.append(distance(base, tip) / max(path_length, 1e-6))
    return np.asarray(values, dtype=np.float32)


def disambiguation_features(points: np.ndarray) -> np.ndarray:
    thumb_tip = points[4]
    index_tip = points[8]
    middle_tip = points[12]
    index_mcp = points[5]
    middle_mcp = points[9]

    index_middle_tip_x_gap = index_tip[0] - middle_tip[0]
    index_middle_tip_y_gap = index_tip[1] - middle_tip[1]
    index_over_middle_cross = index_middle_tip_x_gap * (index_mcp[0] - middle_mcp[0])
    thumb_between_index_middle_x = float(
        min(index_mcp[0], middle_mcp[0]) <= thumb_tip[0] <= max(index_mcp[0], middle_mcp[0])
    )

    return np.asarray(
        [
            index_middle_tip_x_gap,
            index_middle_tip_y_gap,
            index_over_middle_cross,
            thumb_between_index_middle_x,
            distance(thumb_tip, index_mcp),
            distance(thumb_tip, middle_mcp),
            distance(thumb_tip, index_tip),
            distance(thumb_tip, middle_tip),
        ],
        dtype=np.float32,
    )


def distance(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.linalg.norm(first - second))


def joint_angle(first: np.ndarray, middle: np.ndarray, end: np.ndarray) -> float:
    a = first - middle
    b = end - middle
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator < 1e-6:
        return 0.0
    cosine = float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))
    return float(np.arccos(cosine) / np.pi)
