#!/usr/bin/env python3
"""Run streaming RGB video -> YOLO bbox -> HMR2 features -> RGB2Robo motion."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "outputs/streaming"))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--yolo_ckpt", default=str(PROJECT_ROOT / "stream_infer/assets/yolo11s.onnx"))
    parser.add_argument("--hmr2_ckpt", default=None)
    parser.add_argument("--checkpoint", default=str(PROJECT_ROOT / "assets/checkpoints/rgb2robo_multirobot_clean.pth"))
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config/inference.yaml"))
    parser.add_argument("--xml_path", default=str(PROJECT_ROOT / "assets/robot/unitree_g1/g1_mocap_29dof.xml"))
    parser.add_argument("--feature_batch_size", type=int, default=16)
    parser.add_argument("--bbox_batch_size", type=int, default=16)
    parser.add_argument("--lag_mode", choices=["causal", "bidirectional"], default="causal")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing streaming outputs.")
    return parser.parse_args()


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    """Reject output collisions before loading models or processing video."""
    output_names = (
        "bbox_stream.npy", "vit_features.pt", "stream_benchmark.json",
        "g1_raw_pred.npz", "g1_export_meta.json", "stream_meta.json",
    )
    existing = [name for name in output_names if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Streaming outputs already exist in {output_dir}: {', '.join(existing)}. "
            "Use --overwrite to replace them, or choose another --output_dir."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    video_path = Path(args.video_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    prepare_output_dir(output_dir, args.overwrite)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch

    from stream_infer.streaming_pipeline import (
        BBoxSmoother,
        HMR2FeatureStream,
        StreamMotionRunner,
        YoloBBoxStream,
        elapsed_since,
        iter_video_frames,
        now,
        read_video_info,
        save_stream_predictions,
        sync_cuda,
    )

    info = read_video_info(video_path)
    yolo = YoloBBoxStream(args.yolo_ckpt, device="cuda", conf=args.conf)
    feature_stream = HMR2FeatureStream(args.hmr2_ckpt, device="cuda", batch_size=args.feature_batch_size)
    motion_runner = StreamMotionRunner(args.config, args.checkpoint, device="cuda")
    smoother = BBoxSmoother(ema=0.7)

    frame_buffer: List[np.ndarray] = []
    frame_id_buffer: List[int] = []
    bbox_records: List[Dict[str, object]] = []
    feature_chunks: List[torch.Tensor] = []
    per_stage: List[Dict[str, object]] = []

    t_total = now()
    for frame_id, frame in iter_video_frames(video_path):
        if args.max_frames is not None and frame_id >= int(args.max_frames):
            break
        frame_buffer.append(frame)
        frame_id_buffer.append(frame_id)

        if len(frame_buffer) < args.bbox_batch_size:
            continue

        bbox_batch, bbox_time = yolo.detect_batch(frame_buffer, frame_id_buffer)
        bbox_records.extend(
            {
                "frame_id": rec.frame_id,
                "xyxy": list(rec.xyxy),
                "score": rec.score,
                "source": rec.source,
            }
            for rec in bbox_batch
        )
        per_stage.append({"stage": bbox_time.stage, "sec": bbox_time.sec, **bbox_time.extra})

        smoothed_bboxes: List[np.ndarray] = []
        for rec in bbox_batch:
            smoothed_bboxes.append(smoother.update(np.asarray(rec.xyxy, dtype=np.float32)))

        feats, feat_timing = feature_stream.extract_batch(frame_buffer, smoothed_bboxes)
        feature_chunks.append(feats)
        per_stage.append({"stage": "hmr2_prep", "sec": feat_timing["prep_sec"], "batch": len(frame_buffer)})
        per_stage.append({"stage": "hmr2_feature", "sec": feat_timing["feature_sec"], "batch": len(frame_buffer)})

        frame_buffer.clear()
        frame_id_buffer.clear()

    if frame_buffer:
        bbox_batch, bbox_time = yolo.detect_batch(frame_buffer, frame_id_buffer)
        bbox_records.extend(
            {
                "frame_id": rec.frame_id,
                "xyxy": list(rec.xyxy),
                "score": rec.score,
                "source": rec.source,
            }
            for rec in bbox_batch
        )
        per_stage.append({"stage": bbox_time.stage, "sec": bbox_time.sec, **bbox_time.extra})
        smoothed_bboxes = [smoother.update(np.asarray(rec.xyxy, dtype=np.float32)) for rec in bbox_batch]
        feats, feat_timing = feature_stream.extract_batch(frame_buffer, smoothed_bboxes)
        feature_chunks.append(feats)
        per_stage.append({"stage": "hmr2_prep", "sec": feat_timing["prep_sec"], "batch": len(frame_buffer)})
        per_stage.append({"stage": "hmr2_feature", "sec": feat_timing["feature_sec"], "batch": len(frame_buffer)})

    if not feature_chunks:
        raise RuntimeError("no features were extracted from input video")
    feature_tensor = torch.cat(feature_chunks, dim=0).float()
    processed_frames = int(feature_tensor.shape[0])
    if args.max_frames is None and processed_frames != int(info["frames"]):
        raise RuntimeError(f"feature/frame mismatch: features={processed_frames}, video={info['frames']}")

    motion, motion_timing = motion_runner.run_windows(feature_tensor, float(info["fps"]), lag_mode=args.lag_mode)
    per_stage.extend({"stage": rec.stage, "sec": rec.sec, **rec.extra} for rec in motion_timing)
    sync_cuda(torch)
    total_sec = elapsed_since(t_total, torch)

    bbox_path = output_dir / "bbox_stream.npy"
    np.save(bbox_path, np.array([[r["frame_id"], *r["xyxy"], r["score"], 0.99, 0.0] for r in bbox_records], dtype=np.float32))
    feature_path = output_dir / "vit_features.pt"
    torch.save(feature_tensor.cpu(), feature_path)

    meta = {
        "video": info,
        "gpu": str(args.gpu),
        "frames_processed": processed_frames,
        "lag_mode": args.lag_mode,
        "feature_batch_size": int(args.feature_batch_size),
        "bbox_batch_size": int(args.bbox_batch_size),
        "yolo_ckpt": args.yolo_ckpt,
        "encoder_source_root": str(feature_stream.encoder_source_root),
        "hmr2_ckpt": feature_stream.hmr2_ckpt,
        "checkpoint": args.checkpoint,
        "config": args.config,
        "total_sec": total_sec,
        "stages": per_stage,
        "bbox_path": str(bbox_path),
        "feature_path": str(feature_path),
    }
    (output_dir / "stream_benchmark.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    save_stream_predictions(
        output_dir,
        fps=float(info["fps"]),
        motion=motion,
        extra_meta={
            "g1_xml_path": args.xml_path,
            "video_path": str(video_path),
            "config_path": args.config,
            "checkpoint_path": args.checkpoint,
            "lag_mode": args.lag_mode,
            "video_frames": processed_frames,
            "video_fps": float(info["fps"]),
        },
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
