from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.config import DATASET_DIR, DATA_DIR, SUPPORTED_IMAGE_EXTENSIONS  # noqa: E402
from app.features import feature_names, normalize_landmarks  # noqa: E402
from app.landmarks import HandLandmarkDetector  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract normalized MediaPipe landmarks from the ASL dataset.")
    parser.add_argument("--dataset", type=Path, default=DATASET_DIR)
    parser.add_argument("--output", type=Path, default=DATA_DIR / "processed" / "landmarks.csv")
    parser.add_argument("--max-per-class", type=int, default=3000, help="Limit images per class; 0 means all.")
    parser.add_argument("--classes", help="Optional comma-separated class names for parallel extraction.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset.exists():
        raise SystemExit(f"Dataset not found: {args.dataset}")

    detector = HandLandmarkDetector(static_image_mode=True)
    if not detector.available:
        raise SystemExit("MediaPipe is unavailable. Install backend requirements with Python 3.10-3.12.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["label", *feature_names()]
    written = skipped = 0

    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        selected_classes = {item.strip().upper() for item in args.classes.split(",")} if args.classes else None
        class_dirs = sorted(path for path in args.dataset.iterdir() if path.is_dir())
        if selected_classes is not None:
            class_dirs = [path for path in class_dirs if path.name.upper() in selected_classes]
        for class_dir in class_dirs:
            images = [path for path in class_dir.iterdir() if path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS]
            if args.max_per_class and len(images) > args.max_per_class:
                images = random.Random(42).sample(images, args.max_per_class)
            for image_path in images:
                frame = cv2.imread(str(image_path))
                if frame is None:
                    skipped += 1
                    continue
                result = detector.detect(frame)
                if not result.hand_detected:
                    skipped += 1
                    continue
                writer.writerow([class_dir.name.upper(), *normalize_landmarks(result.landmarks).tolist()])
                written += 1
            print(f"{class_dir.name}: processed {len(images)} images")

    print(f"Saved {written} samples to {args.output} ({skipped} skipped).")


if __name__ == "__main__":
    main()
