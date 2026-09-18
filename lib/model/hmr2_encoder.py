"""HMR2.0a visual tokens using the upstream ViT and Transformer modules."""

from pathlib import Path

import torch
from torch import nn

from lib.vendor.hmr2.vit import vit
from lib.vendor.hmr2.components.pose_transformer import TransformerDecoder


DEFAULT_HMR2_CKPT = Path(__file__).resolve().parents[2] / "assets/hmr2/epoch=10-step=25000.ckpt"


class HMR2Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = vit(None)
        self.decoder = TransformerDecoder(
            num_tokens=1, token_dim=1, dim=1024, depth=6, heads=8,
            mlp_dim=1024, dim_head=64, dropout=0.0, emb_dropout=0.0,
            norm="layer", context_dim=1280,
        )

    def forward(self, batch):
        images = batch["img"]
        if images.ndim != 4 or images.shape[1:] != (3, 256, 256):
            raise ValueError(f"HMR2 expects [N, 3, 256, 256] crops, got {tuple(images.shape)}")
        patches = self.backbone(images[..., 32:224])
        context = patches.flatten(2).transpose(1, 2)
        query = torch.zeros((len(images), 1, 1), device=images.device)
        return self.decoder(query, context=context).squeeze(1)


def load_hmr2(checkpoint_path=DEFAULT_HMR2_CKPT):
    """Load visual parameters only; the body model and pose readouts are unused."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint["state_dict"]
    weights = {}
    for key, value in state.items():
        if key.startswith("backbone."):
            weights[key] = value
        elif key.startswith("smpl_head.transformer."):
            weights["decoder." + key[len("smpl_head.transformer."):]] = value
    model = HMR2Encoder()
    model.load_state_dict(weights, strict=True)
    return model.eval()
