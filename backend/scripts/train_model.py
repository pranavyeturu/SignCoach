from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.config import DATA_DIR, MODEL_PATH  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the SignCoach landmark classifier.")
    parser.add_argument("--input", type=Path, default=DATA_DIR / "processed" / "landmarks.csv")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--trees", type=int, default=250)
    parser.add_argument("--jobs", type=int, default=2, help="Parallel workers; keep at 2 on an 8 GB laptop.")
    return parser.parse_args()


def load_samples(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 3:
        raise SystemExit("At least two extracted samples are required.")
    return (
        np.asarray([[float(value) for value in row[1:]] for row in rows[1:]], dtype=np.float32),
        np.asarray([row[0] for row in rows[1:]]),
    )


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Processed landmarks not found: {args.input}")
    features, labels = load_samples(args.input)
    train_x, test_x, train_y, test_y = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    model = RandomForestClassifier(
        n_estimators=args.trees, random_state=42, n_jobs=args.jobs, class_weight="balanced_subsample"
    )
    model.fit(train_x, train_y)
    predictions = model.predict(test_x)

    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model)
    metrics_dir = DATA_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    report = classification_report(test_y, predictions, output_dict=True, zero_division=0)
    metrics = {
        "accuracy": accuracy_score(test_y, predictions),
        "samples": int(len(labels)),
        "classes": model.classes_.tolist(),
        "classificationReport": report,
        "confusionMatrix": confusion_matrix(test_y, predictions, labels=model.classes_).tolist(),
    }
    (metrics_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Saved model to {args.model}")


if __name__ == "__main__":
    main()
