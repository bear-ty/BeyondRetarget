"""Export direct G1 motion predictions without legacy reference-motion artifacts."""

import json
import os
from typing import Dict

import numpy as np


def export_g1_predictions(sequence_dir: str, prediction: Dict[str, np.ndarray], xml_path: str) -> Dict[str, str]:
    """Save the predicted root motion and DoFs used by downstream post-processing."""
    os.makedirs(sequence_dir, exist_ok=True)
    fps = int(np.asarray(prediction["fps"]).reshape(-1)[0])
    raw_path = os.path.join(sequence_dir, "g1_raw_pred.npz")
    np.savez_compressed(
        raw_path,
        fps=np.array([fps], dtype=np.int64),
        root_trans=np.asarray(prediction["root_trans"], dtype=np.float32),
        root_rot_quat=np.asarray(prediction["root_rot_quat"], dtype=np.float32),
        root_rot_6d=np.asarray(prediction["root_rot_6d"], dtype=np.float32),
        dof=np.asarray(prediction["dof"], dtype=np.float32),
    )
    meta_path = os.path.join(sequence_dir, "g1_export_meta.json")
    with open(meta_path, "w", encoding="utf-8") as file_obj:
        json.dump(
            {
                "fps": fps,
                "frames": int(np.asarray(prediction["dof"]).shape[0]),
                "g1_dof": int(np.asarray(prediction["dof"]).shape[1]),
                "xml_path": xml_path,
            },
            file_obj,
            ensure_ascii=False,
            indent=2,
        )
    return {"g1_raw_pred": raw_path, "g1_export_meta": meta_path}
