from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "data" / "msasl" / "phrase_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download/cut MS-ASL clips referenced by a SignCoach manifest.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--limit", type=int, default=0, help="Optional total download limit; 0 means all.")
    parser.add_argument("--timeout", type=int, default=45, help="Maximum seconds to spend on one clip.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.manifest.exists():
        raise SystemExit(f"Manifest not found: {args.manifest}")

    rows = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.limit:
        rows = rows[: args.limit]

    downloaded = skipped = failed = 0
    for row in rows:
        output = ROOT / row["video_path"]
        output.unlink(missing_ok=True) if output.suffix == ".part" else None
        if output.exists() and output.stat().st_size > 0:
            skipped += 1
            continue
        output.with_suffix(output.suffix + ".part").unlink(missing_ok=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        section = f"*{row['start_time']}-{row['end_time']}"
        command = [
            sys.executable,
            "-m",
            "yt_dlp",
            "--quiet",
            "--no-warnings",
            "--force-overwrites",
            "--retries",
            "1",
            "--fragment-retries",
            "1",
            "--socket-timeout",
            "10",
            "--download-sections",
            section,
            "--merge-output-format",
            "mp4",
            "-o",
            str(output),
            row["url"],
        ]
        try:
            subprocess.run(command, check=True, timeout=args.timeout)
            downloaded += 1
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            failed += 1
            output.unlink(missing_ok=True)
            output.with_suffix(output.suffix + ".part").unlink(missing_ok=True)
            print(f"Failed: {row['id']} {row['url']}")
    print(f"Downloaded {downloaded}, skipped {skipped}, failed {failed}")


if __name__ == "__main__":
    main()
