#!/usr/bin/env python3
"""Filter robot raw prediction npz files and re-export FK/body files."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.util.g1_export_utils import export_g1_predictions  # noqa: E402
from lib.util.robot_export_utils import export_robot_predictions  # noqa: E402
from lib.robot.robot_spec import load_robot_spec  # noqa: E402

ROBOT_SPEC_PATHS = {
    "g1": "config/robot/g1.yaml",
    "r1": "config/robot/r1.yaml",
    "gr1t1": "config/robot/gr1t1_without_hand.yaml",
    "h1_with_hand": "config/robot/h1_with_hand_without_hand.yaml",
    "gr2v3_8_7_dummy_hand": "config/robot/gr2v3_8_7_dummy_hand.yaml",
    "t1_serial": "config/robot/t1_serial.yaml",
    "atlas_v4": "config/robot/atlas_v4.yaml",
    "tienkung": "config/robot/tienkung.yaml",
}


def _odd_window(length: int, requested: int) -> int:
    window = max(3, int(requested))
    if window % 2 == 0:
        window += 1
    if window > length:
        window = length if length % 2 == 1 else length - 1
    return max(3, window)


def _filter_array(values: np.ndarray, window: int, poly: int) -> np.ndarray:
    if values.shape[0] < 5:
        return values.astype(np.float32, copy=True)
    win = _odd_window(values.shape[0], window)
    order = min(int(poly), win - 1)
    filtered = savgol_filter(values, win, order, axis=0, mode="interp")
    return filtered.astype(np.float32)


def _normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = quat.astype(np.float64, copy=True)
    quat /= np.clip(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-8, None)
    for idx in range(1, quat.shape[0]):
        if float(np.sum(quat[idx - 1] * quat[idx])) < 0.0:
            quat[idx] *= -1.0
    return quat.astype(np.float32)


def _filter_quat_wxyz(quat_wxyz: np.ndarray, window: int, poly: int) -> np.ndarray:
    quat = _normalize_quat_wxyz(quat_wxyz)
    if quat.shape[0] < 5:
        return quat
    quat_xyzw = quat[:, [1, 2, 3, 0]]
    filtered_xyzw = _filter_array(quat_xyzw, window, poly)
    filtered_wxyz = filtered_xyzw[:, [3, 0, 1, 2]]
    return _normalize_quat_wxyz(filtered_wxyz)


def _quat_wxyz_to_rot6d_mte(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
    matrix = Rotation.from_quat(quat_xyzw).as_matrix().astype(np.float32)
    return matrix[:, :2, :].reshape(quat_wxyz.shape[0], 6).astype(np.float32)


def infer_robot_name(raw_path: Path, data) -> str:
    if "robot_name" in data.files:
        return str(np.asarray(data["robot_name"]).reshape(-1)[0])
    if raw_path.name.startswith("g1_") or raw_path.parent.name.startswith("g1"):
        return "g1"
    if raw_path.name.startswith("r1_") or raw_path.parent.name.startswith("r1"):
        return "r1"
    return "g1"


def default_xml_path(robot_name: str, data) -> str:
    if "xml_path" in data.files:
        return str(np.asarray(data["xml_path"]).reshape(-1)[0])
    if robot_name == "r1":
        return str(PROJECT_ROOT / "assets" / "robot" / "unitree_r1" / "r1_mocap.xml")
    spec_path = PROJECT_ROOT / ROBOT_SPEC_PATHS.get(robot_name, f"config/robot/{robot_name}.yaml")
    if spec_path.exists():
        return str(load_robot_spec(str(spec_path), project_root=str(PROJECT_ROOT)).xml_path)
    return str(PROJECT_ROOT / "assets" / "robot" / "unitree_g1" / "g1_mocap_29dof.xml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter robot raw prediction npz and regenerate FK/body exports.")
    parser.add_argument("--input", required=True, help="Path to g1_raw_pred.npz, r1_raw_pred.npz, or *_raw_pred.npz.")
    parser.add_argument("--output_dir", default=None, help="Output directory. Default: sibling directory with suffix.")
    parser.add_argument("--suffix", default="_filtered", help="Default output directory suffix when --output_dir is omitted.")
    parser.add_argument("--robot", default=None, help="Robot name. Inferred from npz by default.")
    parser.add_argument("--xml_path", default=None, help="Override MuJoCo XML path.")
    parser.add_argument("--root_window", type=int, default=15)
    parser.add_argument("--rot_window", type=int, default=15)
    parser.add_argument("--dof_window", type=int, default=11)
    parser.add_argument("--poly", type=int, default=3)
    parser.add_argument("--keep_root", action="store_true", help="Do not filter root translation.")
    parser.add_argument("--keep_rot", action="store_true", help="Do not filter root quaternion.")
    parser.add_argument("--keep_dof", action="store_true", help="Do not filter dof.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_path = Path(args.input)
    data = np.load(raw_path, allow_pickle=True)
    robot_name = args.robot or infer_robot_name(raw_path, data)
    xml_path = args.xml_path or default_xml_path(robot_name, data)
    output_dir = Path(args.output_dir) if args.output_dir else raw_path.parent.with_name(f"{raw_path.parent.name}{args.suffix}")
    output_dir.mkdir(parents=True, exist_ok=True)

    fps = int(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data.files else 30
    root_trans = np.asarray(data["root_trans"], dtype=np.float32)
    root_rot_quat = np.asarray(data["root_rot_quat"], dtype=np.float32)
    dof = np.asarray(data["dof"], dtype=np.float32)

    filtered_root = root_trans if args.keep_root else _filter_array(root_trans, args.root_window, args.poly)
    filtered_quat = root_rot_quat if args.keep_rot else _filter_quat_wxyz(root_rot_quat, args.rot_window, args.poly)
    filtered_dof = dof if args.keep_dof else _filter_array(dof, args.dof_window, args.poly)
    filtered_rot6d = _quat_wxyz_to_rot6d_mte(filtered_quat)

    prediction = {
        "fps": np.array([fps], dtype=np.int64),
        "root_trans": filtered_root.astype(np.float32),
        "root_rot_quat": filtered_quat.astype(np.float32),
        "root_rot_6d": filtered_rot6d.astype(np.float32),
        "dof": filtered_dof.astype(np.float32),
    }
    if robot_name == "g1":
        exports = export_g1_predictions(str(output_dir), prediction, xml_path)
    else:
        spec_path = PROJECT_ROOT / ROBOT_SPEC_PATHS.get(robot_name, f"config/robot/{robot_name}.yaml")
        model_to_xml_dof = None
        neutral_dof = None
        if Path(spec_path).exists():
            spec = load_robot_spec(str(spec_path), project_root=str(PROJECT_ROOT))
            model_to_xml_dof = spec.model_to_xml_dof
            neutral_dof = spec.neutral_dof
        exports = export_robot_predictions(str(output_dir), prediction, xml_path, robot_name=robot_name, model_to_xml_dof=model_to_xml_dof, neutral_dof=neutral_dof)

    meta = {
        "input": str(raw_path),
        "output_dir": str(output_dir),
        "robot": robot_name,
        "xml_path": xml_path,
        "fps": fps,
        "frames": int(root_trans.shape[0]),
        "root_window": int(args.root_window),
        "rot_window": int(args.rot_window),
        "dof_window": int(args.dof_window),
        "poly": int(args.poly),
        "exports": exports,
    }
    meta_path = output_dir / "filter_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
