from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from postprocess.contact_head import ContactHeadConfig, FootContactPredictor


def load_contact_head(checkpoint_path: str, device: torch.device) -> FootContactPredictor:
    payload = torch.load(checkpoint_path, map_location="cpu")
    cfg_dict = payload.get("contact_head_config", {})
    cfg = ContactHeadConfig(**cfg_dict)
    model = FootContactPredictor(cfg)
    state = payload.get("contact_head_state_dict")
    if state is None:
        state = payload.get("model_state_dict", payload)
        if any(key.startswith("contact_predictor.") for key in state):
            state = {
                key[len("contact_predictor.") :]: value
                for key, value in state.items()
                if key.startswith("contact_predictor.")
            }
    if not isinstance(state, dict) or not state or not all(torch.is_tensor(value) for value in state.values()):
        raise ValueError(
            f"Unsupported contact checkpoint at {checkpoint_path}: "
            "expected contact_head_state_dict or contact_predictor.* entries in model_state_dict"
        )
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_contact_prob(
    model: FootContactPredictor,
    motion_feature: torch.Tensor,
    motion: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    logits = model(motion_feature, motion, valid_mask=valid_mask)
    prob = torch.sigmoid(logits)
    label = (prob > 0.5).to(torch.int64)
    return {"logits": logits, "prob": prob, "label": label}


def save_contact_outputs(output_dir: str, result: Dict[str, torch.Tensor], meta: Dict[str, object]) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "contact_logits.npy", result["logits"].detach().cpu().numpy().astype(np.float32))
    np.save(out / "contact_prob.npy", result["prob"].detach().cpu().numpy().astype(np.float32))
    np.save(out / "contact_label.npy", result["label"].detach().cpu().numpy().astype(np.int64))
    (out / "contact_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
