#!/usr/bin/env python3
import argparse
import os
import os.path as osp
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from lib.util.gen_image_feature import generate_hmr2_feature, list_rgb_frames

from lib.util.video_io import video_info, write_rgb_video
from lib.util.person_track import select_person_track
from lib.util.inference_inputs import load_bbox, load_visual_features

DEFAULT_YOLO_CKPT = osp.join(PROJECT_ROOT, "assets", "yolo", "yolov8x.pt")


def load_rgb_frames(color_dir):
    image_paths = list_rgb_frames(color_dir)

    frames = []
    for image_path in image_paths:
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise FileNotFoundError(f"failed to read image: {image_path}")
        frames.append(bgr[:, :, ::-1])
    return image_paths, np.stack(frames, axis=0)


def image_sequence_to_video(sequence_dir, output_dir, fps, overwrite=False):
    color_dir = osp.join(sequence_dir, "color")
    image_paths, frames = load_rgb_frames(color_dir)
    os.makedirs(output_dir, exist_ok=True)
    video_path = osp.join(output_dir, "_tracker_input.mp4")
    if overwrite or not osp.exists(video_path):
        write_rgb_video(frames, video_path, fps=fps, crf=17)
    return video_path, len(image_paths)


def count_frames_from_video(video_path):
    return video_info(video_path)[0]


def validate_outputs(input_path, bbox_path, feature_path):
    frame_count = (
        len(list_rgb_frames(osp.join(input_path, "color")))
        if osp.isdir(input_path) else count_frames_from_video(input_path)
    )
    bbox = load_bbox(bbox_path, frame_count=frame_count)
    features = load_visual_features(feature_path, frame_count=frame_count)
    return {
        "source_frames": int(frame_count),
        "bbox_shape": tuple(bbox.shape),
        "feature_shape": tuple(features.shape),
    }


def track_person(video_path, yolo_ckpt):
    from ultralytics import YOLO

    frame_count, width, height = video_info(video_path)
    predictions = YOLO(yolo_ckpt).track(
        video_path, device="cuda", conf=0.5, classes=0, verbose=False, stream=True,
    )
    return select_person_track(tqdm(predictions, total=frame_count, desc="YOLO tracking"), frame_count, width, height)


def generate_bbox(
    sequence_dir,
    output_dir,
    fps=30,
    overwrite=False,
    bbox_name="bbox.npy",
    video_path=None,
    yolo_ckpt=None,
):
    os.makedirs(output_dir, exist_ok=True)
    bbox_path = osp.join(output_dir, bbox_name)
    if osp.exists(bbox_path) and not overwrite:
        bbox = load_bbox(bbox_path)
        return bbox_path, bbox

    if video_path:
        video_path = osp.abspath(video_path)
        frame_count = count_frames_from_video(video_path)
    else:
        video_path, frame_count = image_sequence_to_video(sequence_dir, output_dir, fps=fps, overwrite=overwrite)
    yolo_ckpt = osp.abspath(yolo_ckpt or DEFAULT_YOLO_CKPT)
    if not osp.isfile(yolo_ckpt):
        raise FileNotFoundError(
            f"YOLO tracker checkpoint not found: {yolo_ckpt}. "
            "Download it to assets/yolo/yolov8x.pt or pass --yolo_ckpt."
        )
    bbx_xyxy = track_person(video_path, yolo_ckpt).float()
    if int(bbx_xyxy.shape[0]) != int(frame_count):
        raise ValueError(f"Person tracker returned {bbx_xyxy.shape[0]} boxes for {frame_count} frames")

    bbox_wh = bbx_xyxy[:, 2:4] - bbx_xyxy[:, :2]
    if not torch.isfinite(bbx_xyxy).all() or (bbox_wh <= 0).any():
        raise ValueError("Person tracker produced invalid bbox values")

    scores = torch.ones(frame_count, 1, dtype=bbx_xyxy.dtype)
    frame_ids = torch.arange(frame_count, dtype=bbx_xyxy.dtype).view(-1, 1)
    extras = torch.zeros(frame_count, 2, dtype=bbx_xyxy.dtype)
    bbox = torch.cat([frame_ids, bbx_xyxy.cpu(), scores, extras], dim=1).numpy()
    np.save(bbox_path, bbox)
    return bbox_path, bbox


def parse_args():
    parser = argparse.ArgumentParser(description="Generate RGB2Robo bbox tracks and HMR2 visual features.")
    parser.add_argument("--sequence_dir", required=True, help="Source directory containing color/ images.")
    parser.add_argument("--video_path", default=None, help="Optional source video for YOLO tracking and HMR2 features.")
    parser.add_argument("--output_dir", default=None, help="Directory to save bbox.npy and vit_features.pt. Defaults to sequence_dir.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--bbox_name", default="bbox.npy")
    parser.add_argument("--feature_name", default="vit_features.pt")
    parser.add_argument("--yolo_ckpt", default=DEFAULT_YOLO_CKPT, help="YOLOv8x checkpoint path.")
    parser.add_argument("--hmr2_ckpt", default=None, help="HMR2 checkpoint path; defaults to assets/hmr2/epoch=10-step=25000.ckpt.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    sequence_dir = osp.abspath(args.sequence_dir)
    output_dir = osp.abspath(args.output_dir) if args.output_dir else sequence_dir
    bbox_path, bbox = generate_bbox(
        sequence_dir,
        output_dir,
        fps=args.fps,
        overwrite=args.overwrite,
        bbox_name=args.bbox_name,
        video_path=args.video_path,
        yolo_ckpt=args.yolo_ckpt,
    )

    feature_input = osp.abspath(args.video_path) if args.video_path else sequence_dir
    feature_path = generate_hmr2_feature(
        feature_input,
        output_name=args.feature_name,
        color_dir=osp.join(sequence_dir, "color") if not args.video_path else None,
        bbox_path=bbox_path,
        output_dir=output_dir,
        hmr2_ckpt=args.hmr2_ckpt,
        overwrite=args.overwrite,
    )
    validation = validate_outputs(feature_input, bbox_path, feature_path)
    print(
        {
            "bbox_path": bbox_path,
            "feature_path": feature_path,
            "feature_name": args.feature_name,
            "validation": validation,
        }
    )


if __name__ == "__main__":
    main()
