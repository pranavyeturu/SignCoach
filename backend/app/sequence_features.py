from __future__ import annotations

import numpy as np


SEQUENCE_LENGTH = 24
LANDMARK_COUNT = 21


def sequence_feature_names() -> list[str]:
    names = []
    for index in range(LANDMARK_COUNT):
        for axis in ("x", "y"):
            names.extend(
                [
                    f"{axis}{index}_start",
                    f"{axis}{index}_end",
                    f"{axis}{index}_mean",
                    f"{axis}{index}_range",
                    f"{axis}{index}_delta",
                ]
            )
    names.extend(["wrist_path", "wrist_x_range", "wrist_y_range", "frames"])
    return names


def sequence_features(frames: list[list[dict[str, float]]]) -> np.ndarray:
    valid = [frame for frame in frames if len(frame) == LANDMARK_COUNT]
    if len(valid) < 4:
        raise ValueError("At least four landmark frames are required.")

    sampled = resample_frames(valid, SEQUENCE_LENGTH)
    points = np.asarray(
        [[[landmark["x"], landmark["y"]] for landmark in frame] for frame in sampled],
        dtype=np.float32,
    )
    wrist = points[:, 0:1, :]
    points = points - wrist
    scale = float(np.max(np.linalg.norm(points.reshape(-1, 2), axis=1)))
    if scale < 1e-6:
        raise ValueError("Sequence landmarks are too close together.")
    points = points / scale

    start = points[0]
    end = points[-1]
    mean = points.mean(axis=0)
    ranges = np.ptp(points, axis=0)
    delta = end - start
    per_landmark = np.concatenate([start, end, mean, ranges, delta], axis=1).reshape(-1)

    wrist_points = np.asarray([[frame[0]["x"], frame[0]["y"]] for frame in sampled], dtype=np.float32)
    wrist_deltas = np.diff(wrist_points, axis=0)
    wrist_path = float(np.sum(np.linalg.norm(wrist_deltas, axis=1)))
    wrist_stats = np.asarray(
        [
            wrist_path,
            float(np.ptp(wrist_points[:, 0])),
            float(np.ptp(wrist_points[:, 1])),
            float(len(valid)),
        ],
        dtype=np.float32,
    )
    return np.concatenate([per_landmark, wrist_stats]).astype(np.float32)


def resample_frames(frames: list[list[dict[str, float]]], count: int) -> list[list[dict[str, float]]]:
    if len(frames) == count:
        return frames
    indices = np.linspace(0, len(frames) - 1, count).round().astype(int)
    return [frames[index] for index in indices]
