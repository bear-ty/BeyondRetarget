import torch
import torch.nn as nn


class G1Regressor(nn.Module):
    """Lightweight regression head for RGB -> G1 motion."""

    def __init__(self, input_dim=2048, hidden_dim=1024, g1_dof=29):
        super().__init__()
        self.g1_dof = int(g1_dof)
        self.output_dim = 3 + 6 + self.g1_dof

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.drop2 = nn.Dropout()
        self.fc_out = nn.Linear(hidden_dim, self.output_dim)

        nn.init.xavier_uniform_(self.fc_out.weight, gain=0.01)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, x):
        x = self.drop1(torch.relu(self.fc1(x)))
        x = self.drop2(torch.relu(self.fc2(x)))
        return self.fc_out(x)


class SplitResidualG1Regressor(nn.Module):
    """Split root/dof regression head with a small learnable motion prior."""

    def __init__(
        self,
        input_dim=2048,
        hidden_dim=1024,
        g1_dof=29,
        dropout=0.1,
        num_layers=2,
        use_learnable_mean=True,
        root_pos_scale=1.0,
        root_rot_scale=1.0,
        dof_scale=1.0,
        output_gain=0.01,
    ):
        super().__init__()
        self.g1_dof = int(g1_dof)
        self.output_dim = 3 + 6 + self.g1_dof
        self.root_pos_scale = float(root_pos_scale)
        self.root_rot_scale = float(root_rot_scale)
        self.dof_scale = float(dof_scale)

        layers = []
        in_dim = int(input_dim)
        hidden_dim = int(hidden_dim)
        for _ in range(max(1, int(num_layers))):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(float(dropout)),
            ])
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.root_pos_head = nn.Linear(hidden_dim, 3)
        self.root_rot_head = nn.Linear(hidden_dim, 6)
        self.dof_head = nn.Linear(hidden_dim, self.g1_dof)

        if bool(use_learnable_mean):
            self.g1_mean = nn.Parameter(torch.zeros(self.output_dim))
        else:
            self.register_buffer("g1_mean", torch.zeros(self.output_dim))

        for head in (self.root_pos_head, self.root_rot_head, self.dof_head):
            nn.init.xavier_uniform_(head.weight, gain=float(output_gain))
            nn.init.zeros_(head.bias)

    def forward(self, x):
        h = self.trunk(x)
        root_pos = self.root_pos_head(h) * self.root_pos_scale
        root_rot = self.root_rot_head(h) * self.root_rot_scale
        dof = self.dof_head(h) * self.dof_scale
        delta = torch.cat([root_pos, root_rot, dof], dim=-1)
        return self.g1_mean.to(dtype=delta.dtype, device=delta.device) + delta


def build_g1_regressor(input_dim=2048, g1_dof=29, cfg=None):
    cfg = dict(cfg or {})
    head_type = str(cfg.get("type", "mlp")).lower()
    hidden_dim = int(cfg.get("hidden_dim", cfg.get("g1_head_hidden_dim", 1024)))
    if head_type == "split_residual":
        return SplitResidualG1Regressor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            g1_dof=g1_dof,
            dropout=float(cfg.get("dropout", 0.1)),
            num_layers=int(cfg.get("num_layers", 2)),
            use_learnable_mean=bool(cfg.get("use_learnable_mean", True)),
            root_pos_scale=float(cfg.get("root_pos_scale", 1.0)),
            root_rot_scale=float(cfg.get("root_rot_scale", 1.0)),
            dof_scale=float(cfg.get("dof_scale", 1.0)),
            output_gain=float(cfg.get("output_gain", 0.01)),
        )
    if head_type == "mlp":
        return G1Regressor(input_dim=input_dim, hidden_dim=hidden_dim, g1_dof=g1_dof)
    raise ValueError(f"Unsupported G1 head type: {head_type}")
