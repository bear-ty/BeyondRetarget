"""Final visual-temporal encoder used by RGB2Robo."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as functional

from lib.model.cross_window_attention import CrossWindowAttentionBlock
from lib.model.g1_head import build_g1_regressor


def _build_positional_encoding(length: int, dim: int, device: torch.device) -> torch.Tensor:
    position = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1)
    frequency = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32, device=device)
        * (-torch.log(torch.tensor(10000.0, device=device)) / dim)
    )
    encoding = torch.zeros(length, dim, dtype=torch.float32, device=device)
    encoding[:, 0::2] = torch.sin(position * frequency)
    encoding[:, 1::2] = torch.cos(position * frequency)
    return encoding.unsqueeze(0)


class RoPEAttentionG(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float, rope_base: float = 10000.0):
        super().__init__()
        if embed_dim % num_heads or (embed_dim // num_heads) % 2:
            raise ValueError("RoPE attention requires an even per-head dimension.")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.rope_base = rope_base
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = (nn.Linear(embed_dim, embed_dim) for _ in range(4))
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, length, _ = x.shape

        def reshape(proj):
            return proj(x).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

        query, key, value = reshape(self.q_proj), reshape(self.k_proj), reshape(self.v_proj)
        position = torch.arange(length, device=x.device, dtype=torch.float32).unsqueeze(1)
        frequency = 1.0 / (
            float(self.rope_base)
            ** (torch.arange(0, self.head_dim, 2, device=x.device, dtype=torch.float32) / self.head_dim)
        )
        cosine = (position * frequency).cos()[None, None].to(query.dtype)
        sine = (position * frequency).sin()[None, None].to(query.dtype)

        def rotate(tensor: torch.Tensor) -> torch.Tensor:
            result = torch.empty_like(tensor)
            result[..., 0::2] = tensor[..., 0::2] * cosine - tensor[..., 1::2] * sine
            result[..., 1::2] = tensor[..., 0::2] * sine + tensor[..., 1::2] * cosine
            return result

        scores = rotate(query) @ rotate(key).transpose(-2, -1) * (self.head_dim ** -0.5)
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        output = self.attn_dropout(scores.softmax(dim=-1)) @ value
        return self.out_proj(output.transpose(1, 2).contiguous().view(batch, length, self.embed_dim))


class TemporalRefinerBlockG(nn.Module):
    def __init__(self, dim: int, heads: int, ff_mult: int, dropout: float, use_rope_temporal_self_attn: bool = False, rope_base: float = 10000.0):
        super().__init__()
        self.self_norm = nn.LayerNorm(dim)
        self.use_rope_temporal_self_attn = bool(use_rope_temporal_self_attn)
        self.self_attn = (
            RoPEAttentionG(dim, heads, dropout, rope_base)
            if self.use_rope_temporal_self_attn
            else nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        )
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attention_input = self.self_norm(x)
        if self.use_rope_temporal_self_attn:
            attention_output = self.self_attn(attention_input, key_padding_mask=key_padding_mask)
        else:
            attention_output = self.self_attn(
                attention_input,
                attention_input,
                attention_input,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )[0]
        return x + attention_output + self.ff(self.ff_norm(x + attention_output))


class TemporalEncoder(nn.Module):
    """Bidirectional GRU refinement retained by the final visual model."""

    def __init__(self, n_layers: int, input_size: int, hidden_size: int, add_linear: bool, bidirectional: bool, use_residual: bool):
        super().__init__()
        self.input_size = input_size
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, bidirectional=bidirectional, num_layers=n_layers)
        self.linear = nn.Linear(hidden_size * 2, input_size) if bidirectional else (nn.Linear(hidden_size, input_size) if add_linear else None)
        self.use_residual = use_residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, feature_dim = x.shape
        temporal = x.permute(1, 0, 2)
        output, _ = self.gru(temporal)
        if self.linear is not None:
            output = functional.relu(output)
            output = self.linear(output.reshape(-1, output.size(-1))).view(length, batch, feature_dim)
        if self.use_residual and output.shape[-1] == self.input_size:
            output = output + temporal
        return output.permute(1, 0, 2)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in: int = 1024, d_hid: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.w_2(functional.gelu(self.w_1(x)))
        return self.layer_norm(residual + self.dropout(x))


class VisualFeatureAdapter(nn.Module):
    """Normalize and softly adapt per-frame visual tokens before temporal modeling."""

    def __init__(self, input_dim: int, model_dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.1, residual: bool = True, zero_init_last: bool = False):
        super().__init__()
        hidden_dim = int(hidden_dim) if hidden_dim is not None else int(model_dim)
        self.residual = bool(residual)
        self.input_norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(float(dropout))
        self.fc2 = nn.Linear(hidden_dim, model_dim)
        self.drop2 = nn.Dropout(float(dropout))
        self.output_norm = nn.LayerNorm(model_dim)
        self.skip_proj = nn.Linear(input_dim, model_dim) if self.residual and input_dim != model_dim else None
        if bool(zero_init_last):
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        adapted = self.input_norm(x)
        adapted = self.fc2(self.drop1(self.act(self.fc1(adapted))))
        adapted = self.drop2(adapted)
        if self.residual:
            adapted = adapted + (x if self.skip_proj is None else self.skip_proj(x))
        return self.output_norm(adapted)


class RGB2RoboTemporalEncoder(nn.Module):
    """Final HMR2-token encoder with visual temporal and cross-window context."""

    def __init__(
        self,
        enable_cross_window: bool = False,
        cross_window_config: Optional[Dict] = None,
        g1_dof: int = 29,
        input_feature_dim: int = 1024,
        model_feature_dim: int = 1024,
        frontend_config: Optional[Dict] = None,
        g1_head_config: Optional[Dict] = None,
    ):
        super().__init__()
        self.g1_dof = int(g1_dof)
        self.input_feature_dim = int(input_feature_dim)
        self.model_feature_dim = int(model_feature_dim)
        frontend_config = dict(frontend_config or {})
        adapter_cfg = dict(frontend_config.get("visual_adapter", {}) or {})
        self.img_input_proj = (
            VisualFeatureAdapter(
                input_dim=self.input_feature_dim,
                model_dim=self.model_feature_dim,
                hidden_dim=adapter_cfg.get("hidden_dim"),
                dropout=float(adapter_cfg.get("dropout", 0.1)),
                residual=bool(adapter_cfg.get("residual", True)),
                zero_init_last=bool(adapter_cfg.get("zero_init_last", False)),
            )
            if bool(adapter_cfg.get("enabled", False))
            else (nn.Identity() if self.input_feature_dim == self.model_feature_dim else nn.Linear(self.input_feature_dim, self.model_feature_dim))
        )
        self.self_attention = nn.MultiheadAttention(self.model_feature_dim, 4, batch_first=True)
        self.ffn = PositionwiseFeedForward(d_in=self.model_feature_dim, d_hid=self.model_feature_dim, dropout=0.1)
        temporal_cfg = dict(frontend_config.get("temporal_attention", {}) or {})
        self.use_rope_temporal_self_attn = bool(temporal_cfg.get("use_rope", True))
        depth = int(temporal_cfg.get("depth", 0))
        self.temporal_blocks = (
            nn.ModuleList(
                [
                    TemporalRefinerBlockG(
                        dim=self.model_feature_dim,
                        heads=int(temporal_cfg.get("heads", 8)),
                        ff_mult=int(temporal_cfg.get("ff_mult", 4)),
                        dropout=float(temporal_cfg.get("dropout", 0.1)),
                        use_rope_temporal_self_attn=self.use_rope_temporal_self_attn,
                        rope_base=float(temporal_cfg.get("rope_base", 10000.0)),
                    )
                    for _ in range(depth)
                ]
            )
            if bool(temporal_cfg.get("enabled", False)) and depth > 0
            else None
        )
        self.gru = TemporalEncoder(
            n_layers=2,
            input_size=self.model_feature_dim,
            hidden_size=self.model_feature_dim,
            bidirectional=True,
            add_linear=False,
            use_residual=True,
        )
        self.g1_regressor = build_g1_regressor(input_dim=self.model_feature_dim, g1_dof=self.g1_dof, cfg=g1_head_config)
        self.enable_cross_window = bool(enable_cross_window)
        if self.enable_cross_window:
            config = {
                "num_heads": 8,
                "dropout": 0.1,
                "context_span": 8,
                "bidirectional": True,
            }
            config.update(dict(cross_window_config or {}))
            attention_config = dict(config.get("attention", {}) or {})
            num_heads = int(attention_config.get("num_heads", config["num_heads"]))
            dropout = float(attention_config.get("dropout", config["dropout"]))
            bidirectional = bool(attention_config.get("bidirectional", config["bidirectional"]))
            context_span = int(config["context_span"])
            window_length = int(config.get("window_length", 20))
            max_position_length = int(
                config.get("max_position_length", window_length + 2 * context_span)
            )
            self.img_cross_window = CrossWindowAttentionBlock(
                d_model=self.model_feature_dim,
                n_heads=num_heads,
                dropout=dropout,
                context_span=context_span,
                bidirectional=bidirectional,
                max_position_length=max_position_length,
            )
            config.update(
                {
                    "num_heads": num_heads,
                    "dropout": dropout,
                    "bidirectional": bidirectional,
                    "max_position_length": max_position_length,
                }
            )
            self.cross_window_config = config

    def forward(
        self,
        img_feature: torch.Tensor,
        context_kv: Optional[Dict[str, torch.Tensor]] = None,
        window_mask: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        return_motion_feature: bool = False,
        projected_img_feature: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        frame_padding_mask = None if valid_mask is None else ~valid_mask.bool()
        visual = projected_img_feature if projected_img_feature is not None else self.project_visual_feature(img_feature)
        context_kv = self._project_context_kv(context_kv)
        visual = self.gru(visual) + visual
        if self.temporal_blocks is not None:
            if not self.use_rope_temporal_self_attn:
                visual = visual + _build_positional_encoding(visual.shape[1], visual.shape[-1], visual.device)
            for block in self.temporal_blocks:
                visual = block(visual, key_padding_mask=frame_padding_mask)
        else:
            attended, _ = self.self_attention(visual, visual, visual, key_padding_mask=frame_padding_mask)
            visual = visual + attended
        if self.enable_cross_window and context_kv is not None and "img_kv" in context_kv:
            visual = self.img_cross_window(
                x=visual,
                context_kv=context_kv["img_kv"],
                enable_cross_window=True,
                return_kv_cache=False,
                context_mask=window_mask,
            )
        motion_feature = self.ffn(visual).reshape(-1, visual.size(-1))
        if return_motion_feature:
            return motion_feature
        return self.g1_regressor(motion_feature)

    def _project_context_kv(self, context_kv: Optional[Dict[str, torch.Tensor]]) -> Optional[Dict[str, torch.Tensor]]:
        if not context_kv or "img_kv" not in context_kv:
            return context_kv
        projected = dict(context_kv)
        projected["img_kv"] = self.project_visual_feature(context_kv["img_kv"])
        return projected

    def project_visual_feature(self, img_feature: torch.Tensor) -> torch.Tensor:
        return self.img_input_proj(img_feature)
