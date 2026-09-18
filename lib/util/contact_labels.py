"""Shared validation and normalization for left/right foot-contact labels."""

from __future__ import annotations

from typing import Optional

import numpy as np


def normalize_contact_lr(contact, expected_frames: Optional[int] = None) -> np.ndarray:
    """Return contact labels as finite ``float32 [T, 2]`` left/right values.

    Two channels are used directly. MotionPRO four-channel arrays use columns
    1 and 3 (left/right forefoot); legacy arrays with at least eight channels
    use columns 6 and 7. All column indices are zero-based.
    """

    values = np.asarray(contact)
    if values.ndim != 2:
        raise ValueError(f"contact labels must be a rank-2 [T, C] array, got {values.shape}")
    if expected_frames is not None and int(values.shape[0]) != int(expected_frames):
        raise ValueError(
            f"contact frame count must be {int(expected_frames)}, got {int(values.shape[0])}"
        )
    channel_count = int(values.shape[1])
    if channel_count == 2:
        values = values[:, :2]
    elif channel_count == 4:
        values = values[:, [1, 3]]
    elif channel_count >= 8:
        values = values[:, [6, 7]]
    else:
        raise ValueError(
            f"contact labels must be [T, 2], MotionPRO [T, 4], or legacy [T, C>=8], got {values.shape}"
        )
    values = values.astype(np.float32, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("contact labels contain NaN or Inf")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("contact labels must lie in [0, 1]")
    return np.ascontiguousarray(values)
