from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.config import MSASL_MANIFEST_PATH, PHRASE_FEATURES_PATH  # noqa: E402
from app.landmarks import HandLandmarkDetector  # noqa: E402
from app.sequence_features import sequence_feature_names, sequence_features  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract sequence landmark features from downloaded MS-ASL clips.")
    parser.add_argument("--manifest", type=Path, default=MSASL_MANIFEST_PATH)
    parser.add_argument("--output", type=Path, default=PHRASE_FEATURES_PATH)
    parser.add_argument("--max-per-gloss", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.manifest.exists():
        raise SystemExit(f"Manifest not found: {args.manifest}")

    detector = HandLandmarkDetector(static_image_mode=False)
    rows = json.loads(args.manifest.read_text(encoding="utf-8"))
    counts: dict[str, int] = {}
    written = skipped = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", *sequence_feature_names()])
        for row in rows:
            gloss = row["gloss"]
            if counts.get(gloss, 0) >= args.max_per_gloss:
                continue
            video_path = ROOT / row["video_path"]
            if not video_path.exists():
                skipped += 1
                continue
            frames = extract_landmark_frames(video_path, detector)
            if len(frames) < 4:
                skipped += 1
                continue
            writer.writerow([gloss, *sequence_features(frames).tolist()])
            counts[gloss] = counts.get(gloss, 0) + 1
            written += 1
    print(f"Saved {written} sequence samples to {args.output} ({skipped} skipped).")
    for gloss, count in sorted(counts.items()):
        print(f"{gloss}: {count}")


def extract_landmark_frames(video_path: Path, detector: HandLandmarkDetector) -> list[list[dict[str, float]]]:
    capture = cv2.VideoCapture(str(video_path))
    frames = []
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % 2 == 0:
            result = detector.detect(frame)
            if result.hand_detected:
                frames.append(result.landmarks)
        frame_index += 1
    capture.release()
    return frames


if __name__ == "__main__":
    main()
