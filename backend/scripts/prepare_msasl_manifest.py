from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.config import MSASL_DIR, MSASL_MANIFEST_PATH  # noqa: E402


DEFAULT_GLOSSES = ["yes", "no", "good", "morning", "how", "you", "me", "fine", "hello", "what", "name"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a small MS-ASL manifest for SignCoach phrase modeling.")
    parser.add_argument("--msasl-dir", type=Path, default=MSASL_DIR)
    parser.add_argument("--output", type=Path, default=MSASL_MANIFEST_PATH)
    parser.add_argument("--glosses", default=",".join(DEFAULT_GLOSSES))
    parser.add_argument("--max-per-gloss", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    glosses = {item.strip().lower() for item in args.glosses.split(",") if item.strip()}
    manifest = []
    for split in ("train", "val", "test"):
        path = args.msasl_dir / f"MSASL_{split}.json"
        if not path.exists():
            raise SystemExit(f"Missing MS-ASL annotation file: {path}")
        rows = json.loads(path.read_text(encoding="utf-8"))
        counts: dict[str, int] = {}
        for row in rows:
            gloss = str(row.get("clean_text") or row.get("text") or "").lower()
            if gloss not in glosses:
                continue
            if counts.get(gloss, 0) >= args.max_per_gloss:
                continue
            index = counts.get(gloss, 0)
            counts[gloss] = index + 1
            manifest.append(
                {
                    "id": f"{split}_{gloss}_{index:04d}",
                    "split": split,
                    "gloss": gloss,
                    "label": int(row["label"]),
                    "url": normalize_url(str(row["url"])),
                    "start_time": float(row["start_time"]),
                    "end_time": float(row["end_time"]),
                    "file": row.get("file", ""),
                    "video_path": f"data/msasl/videos/{split}/{gloss}/{split}_{gloss}_{index:04d}.mp4",
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(manifest)} manifest rows to {args.output}")
    for gloss in sorted(glosses):
        total = sum(1 for item in manifest if item["gloss"] == gloss)
        print(f"{gloss}: {total}")


def normalize_url(url: str) -> str:
    if url.startswith("http://") or url.startswith("https://"):
        return url
    return f"https://{url}"


if __name__ == "__main__":
    main()
