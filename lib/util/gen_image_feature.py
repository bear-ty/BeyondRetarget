"""Extract the HMR2 visual tokens required by RGB2Robo."""

from __future__ import annotations

import glob
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from lib.model.hmr2_encoder import DEFAULT_HMR2_CKPT, load_hmr2
from lib.util.inference_inputs import load_bbox, load_visual_features
from lib.util.hmr2_preprocess import get_batch, get_bbx_xys_from_xyxy


def list_rgb_frames(color_dir):
    """Return the shared frame order for tracking and feature extraction."""
    frame_dir = Path(color_dir)
    paths = sorted(glob.glob(str(frame_dir / "*.png")) + glob.glob(str(frame_dir / "*.jpg")))
    if not paths:
        raise FileNotFoundError(f"No frames found in {frame_dir}")
    return paths


def generate_hmr2_feature(
    input_path: str,
    output_name: str = "vit_features.pt",
    bbox_name: str = "bbox.npy",
    color_dir: str | None = None,
    bbox_path: str | None = None,
    output_dir: str | None = None,
    hmr2_ckpt: str | None = None,
    overwrite: bool = False,
) -> str:
    """Generate one 1024-D HMR2 token per video frame."""
    source = Path(input_path)
    output_directory = Path(output_dir) if output_dir else source
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / output_name
    if output_path.exists() and not overwrite:
        load_visual_features(output_path)
        return str(output_path)

    bbox_file = Path(bbox_path) if bbox_path else source / bbox_name
    bbox = load_bbox(bbox_file)[:, 1:5].astype(np.float32)
    bbx_xys = get_bbx_xys_from_xyxy(torch.from_numpy(bbox), base_enlarge=1.2).float()
    checkpoint = Path(hmr2_ckpt).expanduser().resolve() if hmr2_ckpt else DEFAULT_HMR2_CKPT
    if not checkpoint.is_file():
        raise FileNotFoundError(f"HMR2 checkpoint not found: {checkpoint}")
    extractor = load_hmr2(checkpoint_path=checkpoint).cuda().eval()

    if source.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}:
        images, _ = get_batch(str(source), bbx_xys, img_ds=0.5)
    else:
        frame_dir = Path(color_dir) if color_dir else source / "color"
        frames = []
        for path in list_rgb_frames(frame_dir):
            frame = cv2.imread(path)
            if frame is None:
                raise ValueError(f"Failed to read image: {path}")
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        images, _ = get_batch(np.stack(frames, axis=0), bbx_xys, img_ds=1.0, path_type="np")

    chunks = []
    with torch.no_grad():
        for start in tqdm(range(0, images.shape[0], 16), desc="HMR2 features"):
            chunks.append(extractor({"img": images[start : start + 16].cuda(non_blocking=True)}).cpu())
    features = torch.cat(chunks, dim=0)
    if features.ndim != 2 or features.shape[1] != 1024 or not torch.isfinite(features).all():
        raise RuntimeError(f"Invalid HMR2 features: {tuple(features.shape)}")
    torch.save(features, output_path)
    return str(output_path)
