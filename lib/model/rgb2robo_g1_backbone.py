"""Final direct RGB/HMR2-feature to robot-motion model."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from lib.model.rgb2robo_temporal_encoder import RGB2RoboTemporalEncoder
from lib.model.robot_motion_temporal_filter import build_robot_motion_temporal_filter


def split_robot_motion_vector(vector: torch.Tensor, robot_dof: int = 29) -> Dict[str, torch.Tensor]:
    return {
        "root_pos": vector[..., :3],
        "root_rot_6d": vector[..., 3:9],
        "dof": vector[..., 9 : 9 + int(robot_dof)],
    }


class RGB2RoboG1Backbone(nn.Module):
    """Direct visual-feature to root translation, root rotation, and DoF model."""

    def __init__(
        self,
        variant: str = "direct",
        g1_dof: int = 29,
        enable_cross_window: bool = True,
        cross_window_config: Optional[Dict] = None,
        input_feature_dim: int = 1024,
        model_feature_dim: int = 1024,
        frontend_config: Optional[Dict] = None,
        g1_head_config: Optional[Dict] = None,
        robot_temporal_filter_config: Optional[Dict] = None,
    ):
        super().__init__()
        if str(variant).lower().replace("-", "_") != "direct":
            raise ValueError("Only the final direct RGB2Robo model variant is supported.")
        self.variant = "direct"
        self.g1_dof = int(g1_dof)
        self.frontend = RGB2RoboTemporalEncoder(
            enable_cross_window=enable_cross_window,
            cross_window_config=cross_window_config or {},
            g1_dof=self.g1_dof,
            input_feature_dim=input_feature_dim,
            model_feature_dim=model_feature_dim,
            frontend_config=frontend_config or {},
            g1_head_config=g1_head_config or {},
        )
        filter_cfg = dict(robot_temporal_filter_config or {})
        self.root_temporal_filter = build_robot_motion_temporal_filter(9, dict(filter_cfg.get("root", filter_cfg) or {}))
        self.dof_temporal_filter = build_robot_motion_temporal_filter(self.g1_dof, dict(filter_cfg.get("dof", filter_cfg) or {}))

    def forward(
        self,
        img_feature: torch.Tensor,
        context_kv: Optional[Dict[str, torch.Tensor]] = None,
        window_mask: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size, sequence_length = img_feature.shape[:2]
        output = self.frontend(
            img_feature,
            context_kv=context_kv,
            window_mask=window_mask,
            valid_mask=valid_mask,
        ).view(batch_size, sequence_length, 9 + self.g1_dof)
        raw = split_robot_motion_vector(output, self.g1_dof)
        root = torch.cat([raw["root_pos"], raw["root_rot_6d"]], dim=-1)
        dof = raw["dof"]
        if self.root_temporal_filter is not None:
            root = self.root_temporal_filter(root, valid_mask=valid_mask)
        if self.dof_temporal_filter is not None:
            dof = self.dof_temporal_filter(dof, valid_mask=valid_mask)
        motion = torch.cat([root[..., :3], root[..., 3:9], dof], dim=-1)
        return {"motion": motion, "raw_motion": output, **split_robot_motion_vector(motion, self.g1_dof)}
