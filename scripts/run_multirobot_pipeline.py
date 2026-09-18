#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.util.window_inference import (  # noqa: E402
    build_context_kv,
    build_window_weight,
    compute_window_starts,
    to_plain_container,
)
from lib.model.rgb2robo_g1_backbone import RGB2RoboG1Backbone  # noqa: E402
from lib.model.rgb2multi_robot_direct import RGB2MultiRobotDirect  # noqa: E402
from lib.robot.robot_spec import load_robot_spec  # noqa: E402
from lib.util.g1_export_utils import export_g1_predictions  # noqa: E402
from lib.util.robot_rotation import rot6d_to_quat_wxyz  # noqa: E402
from lib.util.robot_export_utils import export_robot_predictions  # noqa: E402
from lib.util.inference_inputs import load_visual_features
from lib.util.visual_feature_config import resolve_visual_feature_cfg  # noqa: E402
from postprocess.contact_infer import load_contact_head, predict_contact_prob, save_contact_outputs  # noqa: E402
from postprocess.final_robot_postprocess import (  # noqa: E402
    FINAL_ROBOT_POSTPROCESS_METHOD,
    final_robot_postprocess,
)


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

DEFAULT_BATCH_WINDOWS = 128


def robot_spec_paths() -> Dict[str, str]:
    paths = dict(ROBOT_SPEC_PATHS)
    override_raw = os.environ.get("RGB2ROBO_ROBOT_SPEC_OVERRIDES", "").strip()
    if override_raw:
        overrides = json.loads(override_raw)
        paths.update({str(key): str(value) for key, value in overrides.items()})
    return paths


def remove_prefix(text: str, prefix: str) -> str:
    return text[len(prefix):] if text.startswith(prefix) else text


def load_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state, dict) or not state or not all(torch.is_tensor(value) for value in state.values()):
        checkpoint_type = checkpoint.get("checkpoint_type", "unknown") if isinstance(checkpoint, dict) else "unknown"
        raise ValueError(
            f"Unsupported checkpoint format {checkpoint_type!r} at {checkpoint_path}. "
            "Expected a current RGB2Robo checkpoint containing model_state_dict."
        )
    if any(key.startswith("module.") for key in state):
        state = {remove_prefix(key, "module."): value for key, value in state.items()}
    if any(key.startswith("base_model.") for key in state):
        state = {
            remove_prefix(key, "base_model."): value
            for key, value in state.items()
            if key.startswith("base_model.")
        }
    return state


def checkpoint_robot_heads(checkpoint_path: str, requested: Optional[Iterable[str]] = None) -> Dict[str, Dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    heads = dict(checkpoint.get("robot_heads") or {})
    state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state, dict) or not state or not all(torch.is_tensor(value) for value in state.values()):
        checkpoint_type = checkpoint.get("checkpoint_type", "unknown") if isinstance(checkpoint, dict) else "unknown"
        raise ValueError(
            f"Unsupported checkpoint format {checkpoint_type!r} at {checkpoint_path}. "
            "Expected a current RGB2Robo checkpoint containing model_state_dict."
        )
    shared_root_heads = {
        key.split(".")[1]
        for key in state
        if key.startswith("robot_heads.") and ".shared_head." in key
    }
    for key, value in state.items():
        if not key.startswith("robot_heads.") or not key.endswith(".dof_mean"):
            continue
        robot_name = key.split(".")[1]
        heads.setdefault(robot_name, {"dof": int(value.numel()), "share_g1_root": robot_name != "g1"})
        heads[robot_name]["dof"] = int(value.numel())
    for key, value in state.items():
        if not key.startswith("robot_heads.") or not key.endswith(".g1_mean"):
            continue
        robot_name = key.split(".")[1]
        heads.setdefault(robot_name, {"share_g1_root": False})
        heads[robot_name].setdefault("dof", int(value.numel()) - 9)
    for robot_name in shared_root_heads:
        heads.setdefault(robot_name, {"dof": 0, "share_g1_root": True})
        heads[robot_name]["share_g1_root"] = True
    if requested:
        wanted = [str(item) for item in requested]
        if not heads and wanted == ["g1"]:
            heads = {"g1": {"dof": 29, "share_g1_root": False}}
        missing = [item for item in wanted if item not in heads]
        if missing:
            raise ValueError(f"Requested robot heads not found in checkpoint: {missing}")
        heads = {name: heads[name] for name in wanted}
    missing_dof = [name for name, meta in heads.items() if "dof" not in meta]
    if missing_dof:
        raise ValueError(f"Checkpoint robot head metadata is missing dof for: {missing_dof}")
    return heads


def get_task_cfg(cfg):
    return cfg.task if "task" in cfg else cfg


def load_unified_model(config_path: str, checkpoint_path: str, device: torch.device, robot_heads: Dict[str, Dict[str, object]]) -> RGB2MultiRobotDirect:
    cfg = OmegaConf.load(config_path)
    task = get_task_cfg(cfg)
    cross_window_cfg = to_plain_container(task.get("cross_window_config", {}))
    frontend_cfg = to_plain_container(task.get("frontend", {}))
    g1_head_cfg = to_plain_container(task.get("g1_head", {}))
    robot_temporal_filter_cfg = to_plain_container(task.get("robot_temporal_filter", {}))
    visual_cfg = resolve_visual_feature_cfg(to_plain_container(task.get("visual_feature", {})))
    model_feature_dim = int(visual_cfg["model_dim"])
    base_model = RGB2RoboG1Backbone(
        variant="direct",
        g1_dof=29,
        enable_cross_window=bool(task.get("enable_cross_window", True)),
        cross_window_config=cross_window_cfg,
        input_feature_dim=int(visual_cfg.get("input_dim", 1024)),
        model_feature_dim=model_feature_dim,
        frontend_config=frontend_cfg,
        g1_head_config=g1_head_cfg,
        robot_temporal_filter_config=robot_temporal_filter_cfg,
    )
    model_heads = {}
    for robot_name, meta in robot_heads.items():
        model_heads[robot_name] = {
            "dof": int(meta["dof"]),
            "input_dim": model_feature_dim,
            "share_g1_root": bool(meta.get("share_g1_root", robot_name != "g1")),
            "head_config": g1_head_cfg,
        }
    model = RGB2MultiRobotDirect(base_model, model_heads, robot_temporal_filter_config=robot_temporal_filter_cfg)
    state = load_state_dict(checkpoint_path)
    if set(robot_heads.keys()) == {"g1"} and not any(key.startswith("robot_heads.") for key in state):
        mapped_state = dict(state)
        for key, value in state.items():
            if key.startswith("dof_temporal_filter."):
                mapped_state[f"robot_dof_temporal_filters.g1.{key[len('dof_temporal_filter.'):]}"] = value
        state = mapped_state
    missing, unexpected = model.load_state_dict(state, strict=False)
    important_missing = [
        key for key in missing
        if ".shared_head." not in key and not key.startswith("robot_heads.g1.")
    ]
    if important_missing:
        print(f"[warn] missing keys: {important_missing[:20]} ... total={len(important_missing)}")
    unexpected = [
        key
        for key in unexpected
        if not key.startswith("contact_predictor.")
        and not (key.startswith("robot_heads.") and key.split(".")[1] not in robot_heads)
    ]
    if unexpected:
        print(f"[warn] unexpected keys: {unexpected[:20]} ... total={len(unexpected)}")
    model.to(device)
    model.eval()
    return model


def discover_sequence_dirs(input_root: Path, feature_name: str) -> List[Path]:
    return sorted({path.parent for path in input_root.rglob(feature_name) if path.is_file()})


def load_robot_specs(robot_names: Iterable[str]) -> Dict[str, object]:
    specs = {}
    spec_paths = robot_spec_paths()
    for name in robot_names:
        spec_path = spec_paths.get(name, f"config/robot/{name}.yaml")
        if not Path(spec_path).exists():
            raise FileNotFoundError(f"Robot spec not found for {name}: {spec_path}")
        specs[name] = load_robot_spec(spec_path, project_root=str(PROJECT_ROOT))
    return specs


def raw_file_name(robot_name: str) -> str:
    return "g1_raw_pred.npz" if robot_name == "g1" else f"{robot_name}_raw_pred.npz"


def export_prediction(out_dir: Path, robot_name: str, pred_dict: Dict[str, np.ndarray], spec) -> Dict[str, str]:
    if robot_name == "g1":
        return export_g1_predictions(str(out_dir), pred_dict, spec.xml_path)
    if spec.model_to_xml_dof is not None and pred_dict["dof"].shape[-1] != len(spec.model_to_xml_dof):
        raise ValueError(
            f"{robot_name} predicted dof dim {pred_dict['dof'].shape[-1]} "
            f"!= spec model_to_xml_dof len {len(spec.model_to_xml_dof)}"
        )
    return export_robot_predictions(
        str(out_dir),
        pred_dict,
        spec.xml_path,
        robot_name,
        model_to_xml_dof=spec.model_to_xml_dof,
        neutral_dof=spec.neutral_dof,
    )


def infer_one_sequence(
    model: RGB2MultiRobotDirect,
    contact_model,
    feature_path: Path,
    sequence_id: str,
    pred_root: Path,
    contact_root: Path,
    task_cfg,
    robot_specs: Dict[str, object],
    robot_dofs: Dict[str, int],
    fps: int,
    batch_windows: int,
) -> Dict[str, object]:
    device = next(model.parameters()).device
    robot_names = list(robot_specs.keys())
    visual_cfg = resolve_visual_feature_cfg(to_plain_container(task_cfg.get("visual_feature", {})))
    expected_dim = int(visual_cfg["input_dim"])
    feature_tensor = load_visual_features(feature_path, expected_dim=expected_dim)

    seq_len = int(feature_tensor.shape[0])
    window_length = int(task_cfg.get("window_length", 50))
    overlap_frames = int(task_cfg.get("cross_window_config", {}).get("overlap_frames", 10))
    context_span = int(task_cfg.get("cross_window_config", {}).get("context_span", 8))
    bidirectional = bool(task_cfg.get("cross_window_config", {}).get("attention", {}).get("bidirectional", True))
    step_size = max(1, window_length - overlap_frames)
    starts = compute_window_starts(seq_len, window_length, step_size)
    weights = build_window_weight(window_length)
    total_context = context_span * 2 if bidirectional else context_span

    accum = {
        robot_name: {
            "root_pos": np.zeros((seq_len, 3), np.float32),
            "root_rot_6d": np.zeros((seq_len, 6), np.float32),
            "dof": np.zeros((seq_len, int(robot_dofs[robot_name])), np.float32),
        }
        for robot_name in robot_names
    }
    weight_sum = np.zeros((seq_len, 1), dtype=np.float32)
    contact_logits_sum = np.zeros((seq_len, 2), dtype=np.float32)
    contact_prob_sum = np.zeros((seq_len, 2), dtype=np.float32)
    contact_weight_sum = np.zeros((seq_len, 1), dtype=np.float32)

    with torch.no_grad():
        for batch_start in tqdm(range(0, len(starts), batch_windows), desc=f"infer {sequence_id}", leave=False):
            batch_indices = list(range(batch_start, min(len(starts), batch_start + batch_windows)))
            batch_size = len(batch_indices)
            feature_batch = torch.zeros(batch_size, window_length, feature_tensor.shape[-1], dtype=feature_tensor.dtype)
            valid_mask = torch.zeros(batch_size, window_length, dtype=torch.bool)
            lefts = []
            actual_lengths = []
            context_batch = torch.zeros(batch_size, total_context, feature_tensor.shape[-1], dtype=feature_tensor.dtype)
            window_mask = torch.zeros(batch_size, total_context, dtype=torch.bool)

            for row, win_idx in enumerate(batch_indices):
                left = starts[win_idx]
                right = min(seq_len, left + window_length)
                actual = right - left
                feature_batch[row, :actual] = feature_tensor[left:right]
                valid_mask[row, :actual] = True
                lefts.append(left)
                actual_lengths.append(actual)
                context_kv, one_window_mask = build_context_kv(feature_tensor, starts, win_idx, context_span, bidirectional, window_length)
                if context_kv is not None and one_window_mask is not None:
                    context_batch[row] = context_kv["img_kv"][0]
                    window_mask[row] = one_window_mask[0]

            feature_gpu = feature_batch.to(device=device, non_blocking=True)
            valid_gpu = valid_mask.to(device=device, non_blocking=True)
            context_kv = {"img_kv": context_batch.to(device=device, non_blocking=True)}
            window_mask_gpu = window_mask.to(device=device, non_blocking=True)
            motion_feature = model.extract_motion_feature(
                feature_gpu,
                context_kv=context_kv,
                window_mask=window_mask_gpu,
                valid_mask=valid_gpu,
            )
            pred = {
                robot_name: model.predict_from_feature(
                    motion_feature,
                    batch_size,
                    window_length,
                    robot_name,
                    valid_mask=valid_gpu,
                )
                for robot_name in robot_names
            }
            motion_feature_3d = motion_feature.view(batch_size, window_length, -1)
            contact_out = predict_contact_prob(contact_model, motion_feature_3d, feature_gpu, valid_mask=valid_gpu)

            for row, left in enumerate(lefts):
                actual = actual_lengths[row]
                right = left + actual
                frame_weight = weights[:actual, None]
                for robot_name in robot_names:
                    dof = int(robot_dofs[robot_name])
                    out = pred[robot_name]["motion"][row, :actual].detach().cpu().numpy().astype(np.float32)
                    accum[robot_name]["root_pos"][left:right] += out[:, :3] * frame_weight
                    accum[robot_name]["root_rot_6d"][left:right] += out[:, 3:9] * frame_weight
                    accum[robot_name]["dof"][left:right] += out[:, 9: 9 + dof] * frame_weight
                logits = contact_out["logits"][row, :actual].detach().cpu().numpy().astype(np.float32)
                probs = contact_out["prob"][row, :actual].detach().cpu().numpy().astype(np.float32)
                contact_logits_sum[left:right] += logits * frame_weight
                contact_prob_sum[left:right] += probs * frame_weight
                contact_weight_sum[left:right] += frame_weight
                weight_sum[left:right] += frame_weight

    weight_sum = np.clip(weight_sum, 1e-6, None)
    contact_weight_sum = np.clip(contact_weight_sum, 1e-6, None)
    contact_prob = contact_prob_sum / contact_weight_sum
    contact_logits = contact_logits_sum / contact_weight_sum
    contact_label = (contact_prob > 0.5).astype(np.int64)

    out_seq_root = pred_root / sequence_id
    out_seq_root.mkdir(parents=True, exist_ok=True)
    out_contact_root = contact_root / sequence_id
    out_contact_root.mkdir(parents=True, exist_ok=True)

    exports = {}
    for robot_name in robot_names:
        root_pos = accum[robot_name]["root_pos"] / weight_sum
        root_rot_6d = accum[robot_name]["root_rot_6d"] / weight_sum
        dof = accum[robot_name]["dof"] / weight_sum
        root_quat = rot6d_to_quat_wxyz(torch.from_numpy(root_rot_6d)).cpu().numpy().astype(np.float32)
        pred_dict = {
            "fps": np.array([fps], dtype=np.int64),
            "root_trans": root_pos.astype(np.float32),
            "root_rot_quat": root_quat.astype(np.float32),
            "root_rot_6d": root_rot_6d.astype(np.float32),
            "dof": dof.astype(np.float32),
        }
        exports[robot_name] = export_prediction(out_seq_root / robot_name, robot_name, pred_dict, robot_specs[robot_name])

    save_contact_outputs(
        str(out_contact_root),
        {
            "logits": torch.from_numpy(contact_logits),
            "prob": torch.from_numpy(contact_prob),
            "label": torch.from_numpy(contact_label),
        },
        {"seq_name": sequence_id, "feature_path": str(feature_path), "frames": seq_len, "fps": fps},
    )
    return {"seq_name": sequence_id, "frames": seq_len, "exports": exports, "contact_dir": str(out_contact_root)}


def infer_stage_complete(sequence_id: str, roots: Dict[str, Path], robot_names: List[str]) -> bool:
    raw_root = roots["raw"] / sequence_id
    contact_root = roots["contact"] / sequence_id
    if not (contact_root / "contact_label.npy").exists():
        return False
    return all((raw_root / robot_name / raw_file_name(robot_name)).exists() for robot_name in robot_names)


def infer_stage_fresh(sequence_id: str, roots: Dict[str, Path], robot_names: List[str], feature_path: Path) -> bool:
    if not infer_stage_complete(sequence_id, roots, robot_names):
        return False
    targets = [roots["contact"] / sequence_id / "contact_label.npy"]
    targets.extend(roots["raw"] / sequence_id / name / raw_file_name(name) for name in robot_names)
    return min(path.stat().st_mtime_ns for path in targets) >= feature_path.stat().st_mtime_ns


def final_postprocess_one(raw_path: Path, out_dir: Path, robot_name: str, spec, contact_prob: np.ndarray, contact_label: np.ndarray, fps: int) -> Path:
    """Run the canonical final post-process without persisting transient stages."""
    data = np.load(raw_path)
    final = final_robot_postprocess(
        {key: np.asarray(data[key]) for key in data.files},
        spec,
        contact_prob,
        contact_label,
        fps=fps,
    )
    export_prediction(out_dir, robot_name, final, spec)
    return out_dir / raw_file_name(robot_name)


def load_contact_for_ik(contact_dir: Path) -> np.ndarray:
    prob_path = contact_dir / "contact_prob.npy"
    if prob_path.exists():
        return np.load(prob_path).astype(np.float32)
    return np.load(contact_dir / "contact_label.npy").astype(np.float32)


def postprocess_stage_complete(sequence_id: str, roots: Dict[str, Path], robot_names: List[str]) -> bool:
    postprocess_root = roots["postprocess"] / sequence_id
    return all((postprocess_root / robot_name / raw_file_name(robot_name)).exists() for robot_name in robot_names)


def postprocess_stage_fresh(sequence_id: str, roots: Dict[str, Path], robot_names: List[str]) -> bool:
    if not postprocess_stage_complete(sequence_id, roots, robot_names):
        return False
    sources = [roots["contact"] / sequence_id / "contact_label.npy"]
    prob_path = roots["contact"] / sequence_id / "contact_prob.npy"
    if prob_path.exists():
        sources.append(prob_path)
    sources.extend(roots["raw"] / sequence_id / name / raw_file_name(name) for name in robot_names)
    if not all(path.exists() for path in sources):
        return False
    targets = [roots["postprocess"] / sequence_id / name / raw_file_name(name) for name in robot_names]
    return min(path.stat().st_mtime_ns for path in targets) >= max(path.stat().st_mtime_ns for path in sources)


def process_post_steps_for_sequence(
    sequence_id: str,
    roots: Dict[str, Path],
    robot_names: List[str],
    robot_specs,
    fps: int,
    reuse_existing: bool = False,
) -> Dict[str, object]:
    contact_label = np.load(roots["contact"] / sequence_id / "contact_label.npy")
    contact_for_ik = load_contact_for_ik(roots["contact"] / sequence_id)
    for robot_name in robot_names:
        spec = robot_specs[robot_name]
        raw_path = roots["raw"] / sequence_id / robot_name / raw_file_name(robot_name)
        if not reuse_existing:
            final_postprocess_one(
                raw_path,
                roots["postprocess"] / sequence_id / robot_name,
                robot_name,
                spec,
                contact_for_ik,
                contact_label,
                fps,
            )
    return {"postprocess_skipped": bool(reuse_existing)}


def infer_worker_main(payload: dict) -> List[dict]:
    gpu_id = str(payload["gpu_id"])
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(payload["config"])
    task = get_task_cfg(cfg)
    robot_heads = payload["robot_heads"]
    robot_names = payload["robot_names"]
    robot_specs = load_robot_specs(robot_names)
    robot_dofs = {name: int(robot_heads[name]["dof"]) for name in robot_names}
    model = load_unified_model(payload["config"], payload["checkpoint"], device, {name: robot_heads[name] for name in robot_names})
    contact_checkpoint = payload.get("contact_checkpoint") or payload["checkpoint"]
    contact_model = load_contact_head(contact_checkpoint, device)
    visual_cfg = resolve_visual_feature_cfg(to_plain_container(task.get("visual_feature", {})))
    feature_name = str(visual_cfg["filename"])
    roots = {key: Path(value) for key, value in payload["roots"].items()}
    results = []
    for seq_path in payload["seq_dirs"]:
        seq_dir = Path(seq_path)
        sequence_id = str(seq_dir.relative_to(roots["input"]))
        feature_path = seq_dir / feature_name
        if not feature_path.exists():
            continue
        if payload.get("skip_existing") and infer_stage_fresh(sequence_id, roots, robot_names, feature_path):
            results.append({"seq_name": sequence_id, "gpu": gpu_id, "reinferred": False, "infer": {"seq_name": sequence_id, "skipped": True}})
            continue
        manifest = infer_one_sequence(
            model,
            contact_model,
            feature_path,
            sequence_id,
            roots["raw"],
            roots["contact"],
            task,
            robot_specs,
            robot_dofs,
            int(payload["fps"]),
            int(payload["batch_windows"]),
        )
        results.append({"seq_name": sequence_id, "gpu": gpu_id, "reinferred": True, "infer": manifest})
    return results


def postprocess_worker_main(payload: dict) -> List[dict]:
    gpu_id = str(payload["gpu_id"])
    robot_names = payload["robot_names"]
    robot_specs = load_robot_specs(robot_names)
    roots = {key: Path(value) for key, value in payload["roots"].items()}
    force_sequences = set(payload.get("force_postprocess_sequences") or [])
    results = []
    for seq_path in payload["seq_dirs"]:
        seq_dir = Path(seq_path)
        sequence_id = str(seq_dir.relative_to(roots["input"]))
        reuse_existing = (
            payload.get("skip_existing")
            and sequence_id not in force_sequences
            and postprocess_stage_fresh(sequence_id, roots, robot_names)
        )
        result = process_post_steps_for_sequence(
            sequence_id,
            roots,
            robot_names,
            robot_specs,
            int(payload["fps"]),
            reuse_existing=bool(reuse_existing),
        )
        results.append({"seq_name": sequence_id, "gpu": gpu_id, "postprocess": result})
    return results


def split_round_robin(items: List[Path], parts: int) -> List[List[Path]]:
    shards = [[] for _ in range(max(1, parts))]
    for idx, item in enumerate(items):
        shards[idx % len(shards)].append(item)
    return shards


def run_worker_payloads(ctx, worker, payloads: List[dict]) -> List[List[dict]]:
    if not payloads:
        return []
    if len(payloads) == 1:
        return [worker(payloads[0])]
    with ctx.Pool(processes=len(payloads)) as pool:
        return pool.map(worker, payloads)


def file_signature(path: str) -> Dict[str, object]:
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    return {"path": str(resolved), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def build_run_signature(args, robot_names: List[str]) -> Dict[str, object]:
    contact_checkpoint = args.contact_checkpoint or args.checkpoint
    return {
        "config": file_signature(args.config),
        "checkpoint": file_signature(args.checkpoint),
        "contact_checkpoint": file_signature(contact_checkpoint),
        "robots": list(robot_names),
        "fps": int(args.fps),
        "postprocess_method": FINAL_ROBOT_POSTPROCESS_METHOD,
    }


def manifest_signature(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("run_signature")
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unified multi-robot inference, contact prediction, final post-processing.")
    parser.add_argument("--input_root", required=True, help="Root containing sequence directories with vit_features.pt.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config/inference.yaml"), help="Model and inference configuration.")
    parser.add_argument("--checkpoint", required=True, help="Unified multi-robot checkpoint with the contact head.")
    parser.add_argument("--contact_checkpoint", default="", help="Optional checkpoint to load the contact head from; defaults to --checkpoint.")
    parser.add_argument("--output_root", default=str(PROJECT_ROOT / "outputs/inference"))
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--robots", default="g1,r1,gr1t1,h1_with_hand,gr2v3_8_7_dummy_hand,t1_serial,atlas_v4,tienkung")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--batch_windows", type=int, default=DEFAULT_BATCH_WINDOWS)
    parser.add_argument("--phase", choices=("infer", "postprocess", "all"), default="all")
    parser.add_argument("--postprocess_dirname", default="postprocess")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    if not input_root.is_dir():
        raise NotADirectoryError(input_root)
    out_root = Path(args.output_root)
    raw_root = out_root / "raw"
    postprocess_root = out_root / args.postprocess_dirname
    contact_root = out_root / "contact"
    for path in (raw_root, postprocess_root, contact_root):
        path.mkdir(parents=True, exist_ok=True)

    robot_names = [item.strip() for item in args.robots.split(",") if item.strip()]
    robot_heads = checkpoint_robot_heads(args.checkpoint, robot_names)
    gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpu_ids:
        gpu_ids = ["0"]
    cfg_for_features = OmegaConf.load(args.config)
    feature_name = str(resolve_visual_feature_cfg(to_plain_container(get_task_cfg(cfg_for_features).get("visual_feature", {})))["filename"])
    seq_dirs = discover_sequence_dirs(input_root, feature_name)
    if not seq_dirs:
        raise FileNotFoundError(f"No {feature_name} files found under {input_root}")
    if args.limit > 0:
        seq_dirs = seq_dirs[: int(args.limit)]
    run_signature = build_run_signature(args, robot_names)
    infer_skip_existing = bool(args.skip_existing)
    force_all_postprocess = False
    if infer_skip_existing and args.phase in ("infer", "all"):
        if manifest_signature(out_root / "raw_manifest.json") != run_signature:
            infer_skip_existing = False
            print("[cache] inference signature changed or is unavailable; raw/contact outputs will be regenerated")
    if args.skip_existing and args.phase == "postprocess":
        force_all_postprocess = manifest_signature(out_root / "run_manifest.json") != run_signature
        if force_all_postprocess:
            print("[cache] postprocess signature changed or is unavailable; postprocess outputs will be regenerated")
    shards = split_round_robin(seq_dirs, len(gpu_ids))
    roots = {
        "input": str(input_root),
        "raw": str(raw_root),
        "postprocess": str(postprocess_root),
        "contact": str(contact_root),
    }
    payloads = [
        {
            "gpu_id": gpu_ids[idx],
            "seq_dirs": [str(path) for path in shard],
            "config": args.config,
            "checkpoint": args.checkpoint,
            "contact_checkpoint": args.contact_checkpoint,
            "roots": roots,
            "robot_names": robot_names,
            "robot_heads": robot_heads,
            "fps": int(args.fps),
            "batch_windows": int(args.batch_windows),
        }
        for idx, shard in enumerate(shards)
        if shard
    ]
    print(json.dumps({
        "checkpoint": args.checkpoint,
        "config": args.config,
        "output_root": str(out_root),
        "robots": robot_names,
        "gpus": gpu_ids,
        "sequences": len(seq_dirs),
        "phase": args.phase,
        "batch_windows": int(args.batch_windows),
        "postprocess_method": FINAL_ROBOT_POSTPROCESS_METHOD,
    }, ensure_ascii=False, indent=2))

    ctx = mp.get_context("spawn")

    if args.phase in ("infer", "all"):
        infer_payloads = [
            {
                **payload,
                "skip_existing": infer_skip_existing,
            }
            for payload in payloads
        ]
        infer_worker_results = run_worker_payloads(ctx, infer_worker_main, infer_payloads)
        reinferred_sequences = {
            item["seq_name"]
            for worker_out in infer_worker_results
            for item in worker_out
            if item.get("reinferred")
        }
        infer_manifest = {"sequences": []}
        for worker_out in infer_worker_results:
            for item in worker_out:
                infer_manifest["sequences"].append(item["infer"])
        infer_manifest.update({
            "input_root": str(input_root),
            "checkpoint": args.checkpoint,
            "contact_checkpoint": args.contact_checkpoint or args.checkpoint,
            "config": args.config,
            "robots": robot_names,
            "gpus": gpu_ids,
            "phase": "infer",
            "run_signature": run_signature,
        })
        (out_root / "raw_manifest.json").write_text(json.dumps(infer_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"infer stage done: saved raw/contact to {out_root}")

    if args.phase in ("postprocess", "all"):
        if args.phase == "postprocess":
            reinferred_sequences = {str(seq.relative_to(input_root)) for seq in seq_dirs} if force_all_postprocess else set()
        post_payloads = [
            {
                **payload,
                "skip_existing": args.skip_existing,
                "force_postprocess_sequences": sorted(reinferred_sequences),
            }
            for payload in payloads
        ]
        post_worker_results = run_worker_payloads(ctx, postprocess_worker_main, post_payloads)
        manifest = {"sequences": [
            {"seq_name": item["seq_name"], "gpu": item["gpu"], **item["postprocess"]}
            for worker_out in post_worker_results for item in worker_out
        ]}
        manifest.update({
            "input_root": str(input_root),
            "checkpoint": args.checkpoint,
            "contact_checkpoint": args.contact_checkpoint or args.checkpoint,
            "config": args.config,
            "robots": robot_names,
            "gpus": gpu_ids,
            "phase": args.phase,
            "postprocess_method": FINAL_ROBOT_POSTPROCESS_METHOD,
            "run_signature": run_signature,
        })
        (out_root / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved results to: {out_root}")


if __name__ == "__main__":
    main()
