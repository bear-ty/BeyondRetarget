from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from postprocess.contact_encoder import MotionContactTap, VisualContactReducer


@dataclass(frozen=True)
class ContactHeadConfig:
    motion_dim: int = 1024
    visual_dim: int = 256
    hidden_dim: int = 256
    contact_dim: int = 2
    dropout: float = 0.1


class FootContactHead(nn.Module):
    """Shared left/right foot contact head from trunk motion tokens and compressed visual tokens."""

    def __init__(self, cfg: ContactHeadConfig):
        super().__init__()
        self.cfg = cfg
        input_dim = int(cfg.motion_dim) + int(cfg.visual_dim)
        hidden_dim = int(cfg.hidden_dim)
        self.motion_norm = nn.LayerNorm(int(cfg.motion_dim))
        self.visual_norm = nn.LayerNorm(int(cfg.visual_dim))
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(cfg.dropout)),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(cfg.dropout)),
            nn.Linear(hidden_dim, int(cfg.contact_dim)),
        )

    @staticmethod
    def _reduce_visual_feature(visual_feature: torch.Tensor, target_dim: int) -> torch.Tensor:
        if visual_feature.shape[-1] == target_dim:
            return visual_feature
        if visual_feature.shape[-1] > target_dim:
            return visual_feature[..., :target_dim]
        pad = target_dim - visual_feature.shape[-1]
        zeros = torch.zeros(*visual_feature.shape[:-1], pad, device=visual_feature.device, dtype=visual_feature.dtype)
        return torch.cat([visual_feature, zeros], dim=-1)

    def forward(
        self,
        motion_feature: torch.Tensor,
        visual_feature: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if motion_feature.shape[:2] != visual_feature.shape[:2]:
            raise ValueError(f"motion_feature {motion_feature.shape} and visual_feature {visual_feature.shape} must align")
        if motion_feature.dim() != 3 or visual_feature.dim() != 3:
            raise ValueError("contact head expects [B, T, D] tensors")

        visual_feature = self._reduce_visual_feature(visual_feature, int(self.cfg.visual_dim))
        x = torch.cat([self.motion_norm(motion_feature), self.visual_norm(visual_feature)], dim=-1)
        logits = self.mlp(x)
        if valid_mask is not None:
            logits = logits * valid_mask.unsqueeze(-1).to(logits.dtype)
        return logits


class FootContactPredictor(nn.Module):
    def __init__(self, cfg: ContactHeadConfig):
        super().__init__()
        self.cfg = cfg
        self.motion_tap = MotionContactTap(input_dim=int(cfg.motion_dim), hidden_dim=256, output_dim=int(cfg.motion_dim), dropout=float(cfg.dropout))
        self.visual_reduce = VisualContactReducer(input_dim=1024, hidden_dim=256, output_dim=int(cfg.visual_dim), dropout=float(cfg.dropout))
        self.contact_head = FootContactHead(cfg)

    def forward(self, motion_feature: torch.Tensor, visual_feature: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        motion_token = self.motion_tap(motion_feature.detach())
        visual_token = self.visual_reduce(visual_feature.detach())
        return self.contact_head(motion_token, visual_token, valid_mask=valid_mask)
