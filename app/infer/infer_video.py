#!/usr/bin/env python3
"""Run RGB2Robo directly on a video using HMR2 visual tokens."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.util.gen_bbox_feature import generate_bbox, validate_outputs
from lib.util.gen_image_feature import generate_hmr2_feature


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Input RGB video.")
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "outputs/online"))
    parser.add_argument("--checkpoint", default=str(PROJECT_ROOT / "assets/checkpoints/rgb2robo_multirobot_clean.pth"))
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config/inference.yaml"))
    parser.add_argument("--yolo_ckpt", default=str(PROJECT_ROOT / "assets/yolo/yolov8x.pt"))
    parser.add_argument("--hmr2_ckpt", default=str(PROJECT_ROOT / "assets/hmr2/epoch=10-step=25000.ckpt"))
    parser.add_argument("--robots", default="g1")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--phase", choices=("infer", "postprocess", "all"), default="all")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video = Path(args.video).expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not Path(args.config).is_file():
        raise FileNotFoundError(f"Config not found: {args.config}")

    sample_dir = Path(args.output_dir).expanduser().resolve() / "inputs" / video.stem
    sample_dir.mkdir(parents=True, exist_ok=True)
    bbox_path, _ = generate_bbox(
        sequence_dir=str(sample_dir),
        output_dir=str(sample_dir),
        fps=args.fps,
        overwrite=args.overwrite,
        bbox_name="bbox.npy",
        video_path=str(video),
        yolo_ckpt=args.yolo_ckpt,
    )
    feature_path = generate_hmr2_feature(
        input_path=str(video),
        output_name="vit_features.pt",
        bbox_path=bbox_path,
        output_dir=str(sample_dir),
        hmr2_ckpt=args.hmr2_ckpt,
        overwrite=args.overwrite,
    )
    validate_outputs(str(video), bbox_path, feature_path)

    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/run_multirobot_pipeline.py"),
        "--input_root", str(sample_dir.parent),
        "--config", str(Path(args.config).resolve()),
        "--checkpoint", str(Path(args.checkpoint).resolve()),
        "--output_root", str(Path(args.output_dir).expanduser().resolve() / "predictions"),
        "--gpus", args.gpus,
        "--robots", args.robots,
        "--fps", str(args.fps),
        "--phase", args.phase,
    ]
    if not args.overwrite:
        command.append("--skip_existing")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
