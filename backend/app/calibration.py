from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from app.config import CALIBRATION_LANDMARKS_PATH, METRICS_DIR, MODEL_PATH, PROCESSED_LANDMARKS_PATH
from app.features import feature_names, normalize_landmarks


FEATURE_COLUMNS = feature_names()
CSV_COLUMNS = ["label", *FEATURE_COLUMNS]


def append_calibration_sample(label: str, landmarks: list[dict[str, float]]) -> dict[str, object]:
    normalized_label = label.strip().upper()
    if len(normalized_label) != 1 or not normalized_label.isalpha():
        raise ValueError("Calibration label must be one A-Z letter.")

    features = normalize_landmarks(landmarks)
    CALIBRATION_LANDMARKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    should_write_header = not CALIBRATION_LANDMARKS_PATH.exists()
    with CALIBRATION_LANDMARKS_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if should_write_header:
            writer.writerow(CSV_COLUMNS)
        writer.writerow([normalized_label, *features.tolist()])
    return calibration_summary()


def calibration_summary() -> dict[str, object]:
    counts: Counter[str] = Counter()
    if CALIBRATION_LANDMARKS_PATH.exists():
        with CALIBRATION_LANDMARKS_PATH.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            for row in reader:
                if row:
                    counts[row[0].upper()] += 1
    return {
        "path": str(CALIBRATION_LANDMARKS_PATH),
        "total": sum(counts.values()),
        "byLetter": dict(sorted(counts.items())),
    }


def train_with_calibration(
    base_path: Path = PROCESSED_LANDMARKS_PATH,
    calibration_path: Path = CALIBRATION_LANDMARKS_PATH,
    model_path: Path = MODEL_PATH,
) -> dict[str, object]:
    if not base_path.exists():
        raise FileNotFoundError(f"Base landmark dataset not found: {base_path}")

    features, labels = load_csv_samples(base_path)
    calibration_count = 0
    if calibration_path.exists():
        calibration_features, calibration_labels = load_csv_samples(calibration_path)
        if len(calibration_labels):
            features = np.vstack([features, calibration_features])
            labels = np.concatenate([labels, calibration_labels])
            calibration_count = int(len(calibration_labels))

    class_counts = Counter(labels.tolist())
    if len(class_counts) < 2:
        raise ValueError("Training requires at least two classes.")

    train_x, test_x, train_y, test_y = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels if min(class_counts.values()) >= 2 else None,
    )
    model = RandomForestClassifier(
        n_estimators=250,
        random_state=42,
        n_jobs=2,
        class_weight="balanced_subsample",
    )
    model.fit(train_x, train_y)
    predictions = model.predict(test_x)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {
        "accuracy": float(accuracy_score(test_y, predictions)),
        "samples": int(len(labels)),
        "calibrationSamples": calibration_count,
        "classes": model.classes_.tolist(),
        "classificationReport": classification_report(test_y, predictions, output_dict=True, zero_division=0),
    }
    (METRICS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def load_csv_samples(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2:
        return np.empty((0, len(FEATURE_COLUMNS)), dtype=np.float32), np.asarray([])
    expected_columns = len(FEATURE_COLUMNS) + 1
    if len(rows[0]) != expected_columns:
        raise ValueError(
            f"{path} has {len(rows[0]) - 1} features, but the current model expects {len(FEATURE_COLUMNS)}. "
            "Regenerate landmarks before retraining."
        )
    return (
        np.asarray([[float(value) for value in row[1:]] for row in rows[1:]], dtype=np.float32),
        np.asarray([row[0].upper() for row in rows[1:]]),
    )
