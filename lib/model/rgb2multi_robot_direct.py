from typing import Dict, Optional

import torch
import torch.nn as nn

from lib.model.g1_head import build_g1_regressor
from lib.model.rgb2robo_g1_backbone import RGB2RoboG1Backbone, split_robot_motion_vector
from lib.model.robot_motion_temporal_filter import build_robot_motion_temporal_filter


class SharedRootRobotHead(nn.Module):
    """Share the G1 trunk and root predictors with a robot-specific DoF head."""

    def __init__(self, shared_head: nn.Module, robot_dof: int, output_gain: float = 0.01):
        super().__init__()
        for attr_name in ("trunk", "root_pos_head", "root_rot_head"):
            if not hasattr(shared_head, attr_name):
                raise ValueError("SharedRootRobotHead requires a split_residual style pretrained G1 head")
        self.shared_head = shared_head
        self.g1_dof = int(robot_dof)
        hidden_dim = int(shared_head.dof_head.in_features)
        self.dof_head = nn.Linear(hidden_dim, self.g1_dof)
        self.root_mean = nn.Parameter(torch.zeros(9))
        self.dof_mean = nn.Parameter(torch.zeros(self.g1_dof))
        self.dof_scale = float(getattr(shared_head, "dof_scale", 1.0))
        nn.init.xavier_uniform_(self.dof_head.weight, gain=float(output_gain))
        nn.init.zeros_(self.dof_head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.shared_head.trunk(x)
        root_pos = self.shared_head.root_pos_head(h) * float(getattr(self.shared_head, "root_pos_scale", 1.0))
        root_rot = self.shared_head.root_rot_head(h) * float(getattr(self.shared_head, "root_rot_scale", 1.0))
        root_mean = self.root_mean.to(dtype=x.dtype, device=x.device)
        root = root_mean + torch.cat([root_pos, root_rot], dim=-1)
        dof = self.dof_mean.to(dtype=x.dtype, device=x.device) + self.dof_head(h) * self.dof_scale
        return torch.cat([root, dof], dim=-1)


class RGB2MultiRobotDirect(nn.Module):
    """Direct RGB-to-robot model with a shared visual frontend and robot-specific heads."""

    def __init__(
        self,
        base_model: RGB2RoboG1Backbone,
        robot_heads: Dict[str, Dict],
        robot_temporal_filter_config: Optional[Dict] = None,
    ):
        super().__init__()
        if str(base_model.variant).lower() != "direct":
            raise ValueError("RGB2MultiRobotDirect expects a direct RGB2RoboG1Backbone model")
        self.frontend = base_model.frontend
        self.g1_dof = int(base_model.g1_dof)
        self.robot_dofs = {}
        self.robot_heads = nn.ModuleDict()
        filter_cfg = dict(robot_temporal_filter_config or {})
        root_cfg = dict(filter_cfg.get("root", filter_cfg) or {})
        dof_cfg = dict(filter_cfg.get("dof", filter_cfg) or {})
        self.root_temporal_filter = build_robot_motion_temporal_filter(9, root_cfg)
        self.robot_dof_temporal_filters = nn.ModuleDict()

        for robot_name, head_cfg in robot_heads.items():
            robot_name = str(robot_name)
            dof = int(head_cfg["dof"])
            self.robot_dofs[robot_name] = dof
            self.robot_dof_temporal_filters[robot_name] = build_robot_motion_temporal_filter(dof, dof_cfg) or nn.Identity()
            if robot_name == "g1":
                self.robot_heads[robot_name] = self.frontend.g1_regressor
            elif bool(head_cfg.get("share_g1_root", True)):
                self.robot_heads[robot_name] = SharedRootRobotHead(
                    self.frontend.g1_regressor,
                    dof,
                    output_gain=float(dict(head_cfg.get("head_config", {}) or {}).get("output_gain", 0.01)),
                )
            else:
                full_head = build_g1_regressor(
                    input_dim=int(head_cfg["input_dim"]),
                    g1_dof=dof,
                    cfg=dict(head_cfg.get("head_config", {}) or {}),
                )
                self.robot_heads[robot_name] = full_head

    def forward(
        self,
        img_feature: torch.Tensor,
        robot_name: str = "g1",
        context_kv: Optional[Dict[str, torch.Tensor]] = None,
        window_mask: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        feature = self.extract_motion_feature(
            img_feature,
            context_kv=context_kv,
            window_mask=window_mask,
            valid_mask=valid_mask,
        )
        if isinstance(robot_name, (list, tuple)):
            return {
                str(name): self.predict_from_feature(
                    feature,
                    img_feature.shape[0],
                    img_feature.shape[1],
                    str(name),
                    valid_mask=valid_mask,
                )
                for name in robot_name
            }
        return self.predict_from_feature(
            feature,
            img_feature.shape[0],
            img_feature.shape[1],
            robot_name,
            valid_mask=valid_mask,
        )

    def extract_motion_feature(
        self,
        img_feature: torch.Tensor,
        context_kv: Optional[Dict[str, torch.Tensor]] = None,
        window_mask: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.frontend(
            img_feature,
            context_kv=context_kv,
            window_mask=window_mask,
            valid_mask=valid_mask,
            return_motion_feature=True,
        )

    def predict_from_feature(
        self,
        feature: torch.Tensor,
        batch_size: int,
        seq_len: int,
        robot_name: str = "g1",
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        robot_name = str(robot_name)
        if robot_name not in self.robot_heads:
            raise KeyError(f"Unknown robot head: {robot_name}")
        output = self.robot_heads[robot_name](feature)
        dof = int(self.robot_dofs[robot_name])
        raw_motion = output.view(batch_size, seq_len, 3 + 6 + dof)
        raw_split = split_robot_motion_vector(raw_motion, dof)
        root = torch.cat([raw_split["root_pos"], raw_split["root_rot_6d"]], dim=-1)
        dof_motion = raw_split["dof"]
        if self.root_temporal_filter is not None:
            root = self.root_temporal_filter(root, valid_mask=valid_mask)
        dof_filter = self.robot_dof_temporal_filters[robot_name]
        if not isinstance(dof_filter, nn.Identity):
            dof_motion = dof_filter(dof_motion, valid_mask=valid_mask)
        motion = torch.cat([root[..., :3], root[..., 3:9], dof_motion], dim=-1)
        split = split_robot_motion_vector(motion, dof)
        return {
            "motion": motion,
            "raw_motion": raw_motion,
            **split,
        }
