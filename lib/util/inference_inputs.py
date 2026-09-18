"""Validate the visual features and boxes consumed by inference."""

import numpy as np
import torch


def load_visual_features(path, expected_dim=1024, frame_count=None):
    features = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(features, torch.Tensor)
        or features.layout != torch.strided
        or not features.is_floating_point()
        or features.is_quantized
        or features.ndim != 2
        or features.shape[1] != expected_dim
        or not len(features)
    ):
        raise ValueError(f"Expected a nonempty floating-point feature tensor [T, {expected_dim}]: {path}")
    if not torch.isfinite(features).all():
        raise ValueError(f"Features contain NaN/Inf: {path}")
    if frame_count is not None and len(features) != frame_count:
        raise ValueError(f"feature length mismatch: feature={len(features)}, source={frame_count}, path={path}")
    return features.float()


def load_bbox(path, frame_count=None):
    boxes = np.load(path, allow_pickle=False)
    if boxes.ndim != 2 or boxes.shape[1] != 8 or not len(boxes) or boxes.dtype.kind not in "iuf":
        raise ValueError(f"Expected nonempty numeric boxes [T, 8]: {path}")
    if not np.isfinite(boxes).all() or np.any(boxes[:, 3:5] <= boxes[:, 1:3]):
        raise ValueError(f"Boxes must be finite and have positive width and height: {path}")
    if not np.array_equal(boxes[:, 0], np.arange(len(boxes))):
        raise ValueError(f"Box frame IDs must run from 0 to T-1 in order: {path}")
    if frame_count is not None and len(boxes) != frame_count:
        raise ValueError(f"bbox length mismatch: bbox={len(boxes)}, source={frame_count}, path={path}")
    return boxes
