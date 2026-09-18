from __future__ import annotations

import torch
import torch.nn as nn


class VisualContactReducer(nn.Module):
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 256, output_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected [B, T, D], got {x.shape}")
        return self.net(x)


class MotionContactTap(nn.Module):
    """A narrow motion token tap for contact prediction."""

    def __init__(self, input_dim: int = 1024, hidden_dim: int = 256, output_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected [B, T, D], got {x.shape}")
        return self.net(x)
