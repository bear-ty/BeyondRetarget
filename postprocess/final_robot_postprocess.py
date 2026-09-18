"""Canonical offline robot postprocess: strong filtering plus smooth contact IK."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch

from lib.robot.robot_kinematics import RobotKinematicsModel
from lib.robot.robot_spec import RobotSpec
from lib.util.contact_labels import normalize_contact_lr
from postprocess.postprocess_robot_motion import _filter_array, _filter_quat_wxyz, _quat_wxyz_to_rot6d_mte
from postprocess.robot_ik_postprocess import (
    FootIKConfig,
    _foot_joint_specs,
    _marker_positions,
    apply_root_nonpenetration,
    apply_root_support_grounding,
    foot_support_heights,
    refine_robot_motion_with_contact,
)


FINAL_ROBOT_POSTPROCESS_METHOD = "canonical_contact_filter_ik_nonpenetration_v2"


@dataclass(frozen=True)
class FinalRobotPostprocessConfig:
    pre_root_window: int = 21
    pre_rot_window: int = 21
    pre_dof_window: int = 17
    post_root_window: int = 13
    post_rot_window: int = 13
    post_dof_window: int = 9
    poly: int = 3
    external_xy_gain: float = 1.0
    external_z_gain: float = 0.75
    external_smooth: int = 9
    external_xy_max_offset: float = 0.04
    external_z_max_offset: float = 0.04
    external_fade: int = 6
    height_axis: int = 2
    nonpenetration_margin: float = 0.003
    nonpenetration_smooth: int = 7


def _fps(data: Dict[str, np.ndarray], fallback: int) -> int:
    return int(np.asarray(data.get("fps", np.array([fallback]))).reshape(-1)[0])


def _filter_motion(data: Dict[str, np.ndarray], root_window: int, rot_window: int, dof_window: int, poly: int) -> Dict[str, np.ndarray]:
    quat = _filter_quat_wxyz(np.asarray(data["root_rot_quat"], dtype=np.float32), rot_window, poly)
    return {
        "fps": np.asarray(data["fps"], dtype=np.int64),
        "root_trans": _filter_array(np.asarray(data["root_trans"], dtype=np.float32), root_window, poly),
        "root_rot_quat": quat,
        "root_rot_6d": _quat_wxyz_to_rot6d_mte(quat),
        "dof": _filter_array(np.asarray(data["dof"], dtype=np.float32), dof_window, poly),
    }


def _normalize_contact_label(contact_label: np.ndarray, expected_frames: int) -> np.ndarray:
    return normalize_contact_lr(contact_label, expected_frames=expected_frames) > 0.5


def _foot_marker_series(data: Dict[str, np.ndarray], robot_spec: RobotSpec) -> np.ndarray:
    kinematics = RobotKinematicsModel(robot_spec.xml_path, device="cpu", model_to_xml_dof=robot_spec.model_to_xml_dof)
    foot_ids, foot_offsets = _foot_joint_specs(robot_spec, kinematics)
    root = torch.from_numpy(np.asarray(data["root_trans"], dtype=np.float32))
    quat_wxyz = np.asarray(data["root_rot_quat"], dtype=np.float32)
    dof = torch.from_numpy(np.asarray(data["dof"], dtype=np.float32))
    with torch.no_grad():
        body_pos, body_rot = kinematics.forward_kinematics(root, torch.from_numpy(quat_wxyz[:, [1, 2, 3, 0]]), dof)
        foot = _marker_positions(body_pos, body_rot, foot_ids, torch.as_tensor(foot_offsets, dtype=torch.float32))
    return foot.numpy().astype(np.float32)


def _foot_support_height_series(data: Dict[str, np.ndarray], robot_spec: RobotSpec) -> np.ndarray:
    kinematics = RobotKinematicsModel(robot_spec.xml_path, device="cpu", model_to_xml_dof=robot_spec.model_to_xml_dof)
    foot_ids, foot_offsets = _foot_joint_specs(robot_spec, kinematics)
    root = torch.from_numpy(np.asarray(data["root_trans"], dtype=np.float32))
    quat_wxyz = np.asarray(data["root_rot_quat"], dtype=np.float32)
    dof = torch.from_numpy(np.asarray(data["dof"], dtype=np.float32))
    with torch.no_grad():
        body_pos, body_rot = kinematics.forward_kinematics(root, torch.from_numpy(quat_wxyz[:, [1, 2, 3, 0]]), dof)
        height = foot_support_heights(
            robot_spec, kinematics, body_pos, body_rot, foot_ids, torch.as_tensor(foot_offsets, dtype=torch.float32), 2
        )
    return height.numpy().astype(np.float32)


def _contact_segments(mask: np.ndarray) -> List[Tuple[int, int]]:
    segments: List[Tuple[int, int]] = []
    start = None
    for index, enabled in enumerate(np.concatenate([mask.astype(bool), [False]])):
        if enabled and start is None:
            start = index
        elif start is not None and not enabled:
            if index - start >= 2:
                segments.append((start, index))
            start = None
    return segments


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) == 0:
        return values.astype(np.float32)
    kernel = np.ones(int(window), dtype=np.float32) / float(window)
    padded = np.pad(values, ((window // 2, window - 1 - window // 2), (0, 0)), mode="edge")
    return np.stack([np.convolve(padded[:, dim], kernel, mode="valid") for dim in range(values.shape[-1])], axis=-1).astype(np.float32)


def _fade_weights(length: int, fade: int) -> np.ndarray:
    weights = np.ones(length, dtype=np.float32)
    if fade <= 0 or length <= 1:
        return weights
    ramp = min(int(fade), max(1, length // 2))
    values = np.linspace(1.0 / float(ramp), 1.0, ramp, dtype=np.float32)
    weights[:ramp] = np.minimum(weights[:ramp], values)
    weights[-ramp:] = np.minimum(weights[-ramp:], values[::-1])
    return weights


def _external_root_xyz_offset(foot: np.ndarray, contact: np.ndarray, cfg: FinalRobotPostprocessConfig) -> np.ndarray:
    total = min(len(foot), len(contact))
    foot = np.asarray(foot[:total], dtype=np.float32)
    contact = np.asarray(contact[:total], dtype=bool)
    foot_count = min(foot.shape[1], contact.shape[1])
    height_axis = int(cfg.height_axis)
    horizontal_axes = [index for index in range(3) if index != height_axis]
    desired = np.zeros((total, 3), dtype=np.float32)
    weight = np.zeros((total, 1), dtype=np.float32)
    ground = np.zeros((foot_count,), dtype=np.float32)

    for foot_index in range(foot_count):
        active = contact[:, foot_index]
        ground[foot_index] = (
            float(np.median(foot[active, foot_index, height_axis]))
            if np.any(active)
            else float(np.median(foot[:, foot_index, height_axis]))
        )
        for start, end in _contact_segments(active):
            trajectory = foot[start:end, foot_index]
            offset = np.zeros((end - start, 3), dtype=np.float32)
            offset[:, horizontal_axes] = np.median(trajectory[:, horizontal_axes], axis=0)[None] - trajectory[:, horizontal_axes]
            offset[:, height_axis] = ground[foot_index] - trajectory[:, height_axis]
            fade = _fade_weights(end - start, int(cfg.external_fade))[:, None]
            desired[start:end] += offset * fade
            weight[start:end] += fade

    valid = weight[:, 0] > 1e-6
    offset = np.zeros_like(desired)
    offset[valid] = desired[valid] / weight[valid]
    offset = _moving_average(offset, int(cfg.external_smooth))
    xy_norm = np.linalg.norm(offset[:, horizontal_axes], axis=-1, keepdims=True)
    offset[:, horizontal_axes] *= np.minimum(1.0, float(cfg.external_xy_max_offset) / np.clip(xy_norm, 1e-8, None))
    offset[:, height_axis] = np.clip(offset[:, height_axis], -float(cfg.external_z_max_offset), float(cfg.external_z_max_offset))
    offset[:, horizontal_axes] *= float(cfg.external_xy_gain)
    offset[:, height_axis] *= float(cfg.external_z_gain)
    return offset.astype(np.float32)


def final_robot_postprocess(
    raw_motion: Dict[str, np.ndarray],
    robot_spec: RobotSpec,
    contact_prob: np.ndarray,
    contact_label: np.ndarray,
    fps: int = 30,
    cfg: FinalRobotPostprocessConfig = FinalRobotPostprocessConfig(),
) -> Dict[str, np.ndarray]:
    """Apply the canonical pipeline and return only its final motion."""
    filtered = _filter_motion(raw_motion, cfg.pre_root_window, cfg.pre_rot_window, cfg.pre_dof_window, cfg.poly)
    effective_fps = _fps(filtered, fps)
    frame_count = len(filtered["root_trans"])
    contact_probability = normalize_contact_lr(contact_prob, expected_frames=frame_count)
    refined = refine_robot_motion_with_contact(
        robot_spec,
        filtered["root_trans"],
        filtered["root_rot_quat"],
        filtered["dof"],
        contact_probability,
        robot_spec.xml_path,
        FootIKConfig(height_axis=int(cfg.height_axis), fps=float(effective_fps)),
    )
    post_ik = {
        "fps": filtered["fps"],
        "root_trans": refined["root_pos"].astype(np.float32),
        "root_rot_quat": refined["root_rot_quat_wxyz"].astype(np.float32),
        "dof": refined["dof"].astype(np.float32),
    }
    post_filtered = _filter_motion(
        post_ik,
        cfg.post_root_window,
        cfg.post_rot_window,
        cfg.post_dof_window,
        cfg.poly,
    )
    foot = _foot_marker_series(post_filtered, robot_spec)
    contact = _normalize_contact_label(contact_label, expected_frames=frame_count)
    total = frame_count
    final = dict(post_filtered)
    final["root_trans"] = post_filtered["root_trans"].copy()
    final["root_trans"][:total] += _external_root_xyz_offset(foot[:total], contact[:total], cfg)
    if robot_spec.foot_grounding_meshes:
        support_height = _foot_support_height_series(final, robot_spec)
        final["root_trans"][:total] = apply_root_support_grounding(
            final["root_trans"][:total], contact[:total], support_height[:total], height_axis=int(cfg.height_axis)
        )
    support_height = _foot_support_height_series(final, robot_spec)
    final["root_trans"] = apply_root_nonpenetration(
        final["root_trans"],
        support_height,
        height_axis=int(cfg.height_axis),
        margin=float(cfg.nonpenetration_margin),
        smooth_window=int(cfg.nonpenetration_smooth),
    )
    return final
