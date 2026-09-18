"""Shared windowing utilities for final RGB2Robo inference."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf


def to_plain_container(cfg) -> dict:
    """Resolve an OmegaConf mapping while accepting ordinary dictionaries."""
    if OmegaConf.is_config(cfg):
        return OmegaConf.to_container(cfg, resolve=True)
    return dict(cfg or {})


def compute_window_starts(seq_len: int, window_length: int, step_size: int) -> List[int]:
    if seq_len <= window_length:
        return [0]
    starts = list(range(0, seq_len - window_length + 1, step_size))
    last_start = seq_len - window_length
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def build_window_weight(window_length: int) -> np.ndarray:
    center = (window_length - 1) * 0.5
    distance = np.abs(np.arange(window_length, dtype=np.float32) - center)
    weight = np.ones(window_length, dtype=np.float32) if center <= 0 else 1.0 - distance / center
    return np.clip(weight, 1e-3, None)


def build_context_kv(
    feature_tensor: torch.Tensor,
    starts: List[int],
    current_idx: int,
    context_span: int,
    bidirectional: bool,
    window_length: int,
) -> Tuple[Optional[Dict[str, torch.Tensor]], Optional[torch.Tensor]]:
    seq_len = int(feature_tensor.shape[0])
    past_features = []
    future_features = []
    for window_idx in range(max(0, current_idx - context_span), current_idx):
        left = starts[window_idx]
        past_features.append(feature_tensor[left : min(seq_len, left + window_length)].mean(dim=0))
    if bidirectional:
        for window_idx in range(current_idx + 1, min(len(starts), current_idx + context_span + 1)):
            left = starts[window_idx]
            future_features.append(feature_tensor[left : min(seq_len, left + window_length)].mean(dim=0))
    total_context = context_span * 2 if bidirectional else context_span
    if not past_features and not future_features:
        return None, None
    context = feature_tensor.new_zeros(total_context, feature_tensor.shape[-1])
    mask = torch.zeros(total_context, dtype=torch.bool, device=feature_tensor.device)
    # Attention assigns past slots offsets -context_span..-1 and future slots +1..+context_span.
    if past_features:
        context[context_span - len(past_features) : context_span] = torch.stack(past_features)
        mask[context_span - len(past_features) : context_span] = True
    if future_features:
        context[context_span : context_span + len(future_features)] = torch.stack(future_features)
        mask[context_span : context_span + len(future_features)] = True
    return {"img_kv": context.unsqueeze(0)}, mask.unsqueeze(0)
