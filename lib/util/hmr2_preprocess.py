"""Prepare centered RGB crops for HMR2.0a visual token extraction."""

import cv2
import numpy as np
import torch

from lib.util.video_io import read_rgb_video


def get_bbx_xys_from_xyxy(boxes, base_enlarge=1.2):
    """Convert pixel corners into centers and a square extent covering a 3:4 box."""
    centers = (boxes[..., :2] + boxes[..., 2:]) * 0.5
    dimensions = boxes[..., 2:] - boxes[..., :2]
    extent = torch.maximum(dimensions[..., 1], dimensions[..., 0] / 0.75)
    return torch.cat((centers, (extent * base_enlarge).unsqueeze(-1)), dim=-1)


def prepare_crops(frames, boxes, image_scale=1.0, size=256):
    """Warp square boxes onto a pixel-centered grid, then normalize RGB channels."""
    if not isinstance(frames, np.ndarray) or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("Expected RGB frames [N, H, W, 3]")
    if boxes.shape != (len(frames), 3) or not len(frames):
        raise ValueError("Expected one center/extent box [N, 3] per frame")
    scaled = (boxes.detach().cpu() * image_scale).numpy()
    if not np.isfinite(scaled).all() or np.any(scaled[:, 2] <= 0):
        raise ValueError("Crop boxes must be finite and have positive extents")
    crops = []
    for image, (cx, cy, extent) in zip(frames, scaled):
        blur = extent / size / 2.0
        if blur > 1.1:
            image = cv2.GaussianBlur(image, (5, 5), (blur - 1.0) / 2.0)
        center = np.array([cx, cy], dtype=np.float32)
        half = extent / 2.0
        # Preserve float32 image coordinates before solving the affine map.
        anchors = np.empty((3, 2), dtype=np.float32)
        anchors[0] = center - half
        anchors[1] = center.astype(np.float64) + [half, -half]
        anchors[2] = center
        edge = float(size - 1)
        target = np.array([[0, 0], [edge, 0], [edge / 2, edge / 2]], dtype=np.float32)
        matrix = cv2.getAffineTransform(anchors, target)
        crops.append(cv2.warpAffine(image, matrix, (size, size), flags=cv2.INTER_LINEAR))
    pixels = torch.from_numpy(np.stack(crops)).float().div(255.0)
    normalized = (pixels - pixels.new_tensor([0.485, 0.456, 0.406])) / pixels.new_tensor([0.229, 0.224, 0.225])
    return normalized.permute(0, 3, 1, 2)


def get_batch(input_path, bbx_xys, img_ds=0.5, img_dst_size=256, path_type="video"):
    if path_type == "video":
        frames = read_rgb_video(input_path, scale=img_ds)
    elif path_type == "np" and img_ds == 1.0:
        frames = input_path
    else:
        raise ValueError("Use a video input or unscaled NumPy RGB frames")
    return prepare_crops(frames, bbx_xys, img_ds, img_dst_size), bbx_xys.cpu().clone()
