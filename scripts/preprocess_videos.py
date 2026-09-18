#!/usr/bin/env python3
"""Extract person boxes and HMR2 features from a collection of RGB videos."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.util.gen_bbox_feature import generate_bbox, validate_outputs
from lib.util.gen_image_feature import generate_hmr2_feature


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_root", type=Path, required=True, help="Recursively search this directory for videos.")
    parser.add_argument("--output_root", type=Path, help="Save each video under its relative path without the extension. Defaults to saving beside each video.")
    parser.add_argument("--video_glob", default="1.mp4", help="Video pattern; 1.mp4 selects the front view in MotionPRO.")
    parser.add_argument("--yolo_ckpt", default=str(PROJECT_ROOT / "assets/yolo/yolov8x.pt"))
    parser.add_argument("--hmr2_ckpt", default=str(PROJECT_ROOT / "assets/hmr2/epoch=10-step=25000.ckpt"))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    if not input_root.is_dir():
        raise NotADirectoryError(input_root)
    videos = sorted(video for video in input_root.rglob(args.video_glob) if video.is_file())
    if not videos:
        raise FileNotFoundError(f"No videos matching {args.video_glob!r} found under {input_root}")
    if args.output_root is None:
        destinations = [video.parent for video in videos]
    else:
        output_root = args.output_root.expanduser().resolve()
        destinations = [output_root / video.relative_to(input_root).with_suffix("") for video in videos]
    if len(set(destinations)) != len(destinations):
        raise ValueError("Multiple videos target the same output directory; select one view per sequence or use --output_root with unique video stems.")

    failures = []
    for video, sequence_dir in zip(videos, destinations):
        try:
            bbox_path, _ = generate_bbox(
                sequence_dir=str(sequence_dir), output_dir=str(sequence_dir),
                fps=args.fps, overwrite=args.overwrite, video_path=str(video),
                yolo_ckpt=args.yolo_ckpt,
            )
            feature_path = generate_hmr2_feature(
                input_path=str(video), output_name="vit_features.pt",
                bbox_path=bbox_path, output_dir=str(sequence_dir),
                hmr2_ckpt=args.hmr2_ckpt, overwrite=args.overwrite,
            )
            validation = validate_outputs(str(video), bbox_path, feature_path)
            print(f"prepared {video.relative_to(input_root)} -> {sequence_dir}: {validation}")
        except Exception as error:
            failures.append((video, str(error)))
            print(f"failed {video.relative_to(input_root)}: {error}", file=sys.stderr)
    print(f"prepared={len(videos) - len(failures)} failed={len(failures)}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
