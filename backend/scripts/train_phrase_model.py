from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.config import METRICS_DIR, PHRASE_FEATURES_PATH, PHRASE_MODEL_PATH  # noqa: E402
from app.sequence_features import sequence_feature_names  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small MS-ASL sequence model for SignCoach phrases.")
    parser.add_argument("--input", type=Path, default=PHRASE_FEATURES_PATH)
    parser.add_argument("--model", type=Path, default=PHRASE_MODEL_PATH)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--min-samples-per-class", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features, labels = load_samples(args.input)
    features, labels, dropped = filter_sparse_classes(features, labels, args.min_samples_per_class)
    if len(set(labels.tolist())) < 2:
        raise SystemExit("Need at least two phrase classes to train.")
    class_counts = Counter(labels.tolist())
    can_stratify = min(class_counts.values()) >= 2 and len(labels) >= len(class_counts) * 2
    test_size = max(len(class_counts), int(round(len(labels) * 0.25)))
    if test_size >= len(labels):
        test_size = len(class_counts)
    train_x, test_x, train_y, test_y = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=42,
        stratify=labels if can_stratify else None,
    )
    model = RandomForestClassifier(
        n_estimators=args.trees,
        random_state=42,
        n_jobs=2,
        class_weight="balanced_subsample",
    )
    model.fit(train_x, train_y)
    predictions = model.predict(test_x)
    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model)

    metrics = {
        "accuracy": float(accuracy_score(test_y, predictions)),
        "samples": int(len(labels)),
        "classes": model.classes_.tolist(),
        "classCounts": dict(sorted(class_counts.items())),
        "droppedClasses": dropped,
        "classificationReport": classification_report(test_y, predictions, output_dict=True, zero_division=0),
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "phrase_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Saved phrase model to {args.model}")
    if dropped:
        print(f"Dropped sparse classes: {', '.join(f'{label}={count}' for label, count in dropped.items())}")


def load_samples(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise SystemExit(f"Sequence features not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 4:
        raise SystemExit("At least three sequence samples are required.")
    expected_columns = len(sequence_feature_names()) + 1
    if len(rows[0]) != expected_columns:
        raise SystemExit(f"Expected {expected_columns - 1} features, found {len(rows[0]) - 1}.")
    return (
        np.asarray([[float(value) for value in row[1:]] for row in rows[1:]], dtype=np.float32),
        np.asarray([row[0] for row in rows[1:]]),
    )


def filter_sparse_classes(
    features: np.ndarray, labels: np.ndarray, min_samples: int
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    counts = Counter(labels.tolist())
    keep = {label for label, count in counts.items() if count >= min_samples}
    if len(keep) == len(counts):
        return features, labels, {}
    mask = np.asarray([label in keep for label in labels], dtype=bool)
    dropped = {label: count for label, count in sorted(counts.items()) if label not in keep}
    return features[mask], labels[mask], dropped


if __name__ == "__main__":
    main()
