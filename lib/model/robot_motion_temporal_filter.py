from typing import Optional

import torch
import torch.nn as nn


class RobotTemporalFilterBranch(nn.Module):
    """TemporalFilterFrontEndG-style multi-kernel Conv1d branch for robot vectors."""

    def __init__(
        self,
        channels: int,
        kernel_sizes,
        dropout: float,
        groups: int = 1,
    ):
        super().__init__()
        if not kernel_sizes:
            raise ValueError("kernel_sizes must not be empty for RobotTemporalFilterBranch")
        self.input_norm = nn.LayerNorm(int(channels))
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(
                    int(channels),
                    int(channels),
                    kernel_size=int(kernel_size),
                    padding=int(kernel_size) // 2,
                    groups=int(groups),
                )
                for kernel_size in kernel_sizes
            ]
        )
        self.mix = nn.Sequential(
            nn.GELU(),
            nn.Conv1d(int(channels), int(channels), kernel_size=1, groups=int(groups)),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = self.input_norm(x).transpose(1, 2)
        filtered = sum(branch(x_t) for branch in self.branches) / float(len(self.branches))
        return self.mix(filtered).transpose(1, 2)


class RobotMotionTemporalFilter(nn.Module):
    """Default TemporalFilterFrontEndG-style gated residual filter for robot root/DoF vectors."""

    def __init__(
        self,
        channels: int,
        kernel_sizes=(5, 11, 21),
        dropout: float = 0.1,
        gate_init: float = -1.5,
        groups: int = 1,
    ):
        super().__init__()
        self.channels = int(channels)
        self.branch = RobotTemporalFilterBranch(
            channels=self.channels,
            kernel_sizes=kernel_sizes,
            dropout=float(dropout),
            groups=int(groups),
        )
        self.gate_head = nn.Sequential(
            nn.LayerNorm(self.channels * 2),
            nn.Linear(self.channels * 2, 1),
        )
        self._init_gate_head(float(gate_init))

    def _init_gate_head(self, init_bias: float) -> None:
        last = self.gate_head[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.constant_(last.bias, float(init_bias))

    def forward(self, motion: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if motion.dim() != 3:
            raise ValueError(f"motion must be [B, T, C], got {motion.shape}")
        if motion.shape[-1] != self.channels:
            raise ValueError(f"motion channels must be {self.channels}, got {motion.shape[-1]}")
        delta = self.branch(motion)
        gate_input = torch.cat([motion, delta], dim=-1)
        gate = torch.sigmoid(self.gate_head(gate_input))
        filtered = motion + gate * delta
        if valid_mask is not None:
            filtered = filtered * valid_mask.unsqueeze(-1).to(filtered.dtype)
        return filtered


def build_robot_motion_temporal_filter(
    channels: int,
    cfg: Optional[dict] = None,
) -> Optional[RobotMotionTemporalFilter]:
    cfg = dict(cfg or {})
    if not bool(cfg.get("enabled", False)):
        return None
    return RobotMotionTemporalFilter(
        channels=int(channels),
        kernel_sizes=tuple(cfg.get("kernel_sizes", (5, 11, 21))),
        dropout=float(cfg.get("dropout", 0.1)),
        gate_init=float(cfg.get("gate_init", -1.5)),
        groups=int(cfg.get("groups", 1)),
    )
