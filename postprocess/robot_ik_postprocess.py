from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from lib.robot.robot_kinematics import RobotKinematicsModel
from lib.robot.robot_spec import RobotSpec, load_robot_spec, model_dof_to_xml_dof
from lib.util.geometry import rotation_matrix_to_angle_axis


@dataclass
class FootIKConfig:
    contact_threshold: float = 0.6
    contact_exit_threshold: Optional[float] = 0.35
    min_segment: int = 4
    max_contact_gap: int = 5
    contact_fade: int = 4
    height_axis: int = 2
    enable_contact_motion_gate: bool = True
    contact_motion_speed_threshold_mps: float = 1.0
    fps: float = 30.0
    root_xy_smooth: int = 9
    root_xy_gain: float = 1.0
    root_xy_clip: float = 0.02
    enable_root_xyz_stabilize: bool = True
    root_xyz_xy_clip: float = 0.15
    root_xyz_z_gain: float = 0.75
    root_xyz_z_clip: float = 0.04
    root_xyz_fade: int = 6
    foot_target_smooth: int = 4
    preserve_root_xy: bool = False
    ik_steps: int = 2
    ccd_step_scale: float = 0.10
    ccd_threshold: float = 1e-3
    enable_ik: bool = True
    enable_ik_delta_smoothing: bool = True
    ik_delta_smooth_window: int = 5
    ik_delta_rate_limit: float = 0.18
    enable_contact_foot_flat: bool = True
    contact_foot_flat_gain: float = 0.6
    contact_foot_flat_max_step: float = 0.16
    contact_foot_flat_local_axis: int = 2


def _foot_joint_specs(robot_spec: RobotSpec, kinematics: RobotKinematicsModel) -> Tuple[List[int], List[List[float]]]:
    preferred_names = ["left_foot", "right_foot"]
    wanted = []
    offsets = []
    for name in preferred_names:
        for item in robot_spec.foot_markers:
            if item.name != name:
                continue
            try:
                wanted.append(kinematics.body_names.index(item.robot_link))
                offsets.append([0.0, 0.0, 0.0] if item.local_offset is None else [float(v) for v in item.local_offset])
            except ValueError:
                pass
            break
    if len(wanted) == 2:
        return wanted, offsets
    fallback = []
    fallback_offsets = []
    for item in robot_spec.foot_markers:
        if "foot" in item.name or "toe" in item.robot_link:
            try:
                fallback.append(kinematics.body_names.index(item.robot_link))
                fallback_offsets.append([0.0, 0.0, 0.0] if item.local_offset is None else [float(v) for v in item.local_offset])
            except ValueError:
                continue
    if len(fallback) >= 2:
        return fallback[:2], fallback_offsets[:2]
    raise RuntimeError(f"Could not resolve foot links for {robot_spec.name}")


def _foot_joint_indices(robot_spec: RobotSpec, kinematics: RobotKinematicsModel) -> List[int]:
    foot_ids, _ = _foot_joint_specs(robot_spec, kinematics)
    return foot_ids


def _marker_positions(
    body_pos: torch.Tensor,
    body_rot_xyzw: torch.Tensor,
    body_ids: Sequence[int],
    offsets: torch.Tensor,
) -> torch.Tensor:
    markers = body_pos[:, list(body_ids)]
    if offsets.numel() == 0 or not bool(torch.any(offsets != 0.0)):
        return markers
    marker_rot = _quat_xyzw_to_rotmat(body_rot_xyzw[:, list(body_ids)])
    rotated_offsets = torch.einsum("tfij,fj->tfi", marker_rot, offsets.to(device=markers.device, dtype=markers.dtype))
    return markers + rotated_offsets


@lru_cache(maxsize=None)
def _mesh_convex_vertices(mesh_path: str) -> np.ndarray:
    import trimesh

    return np.asarray(trimesh.load_mesh(mesh_path, process=False).convex_hull.vertices, dtype=np.float32)


def foot_support_heights(
    robot_spec: RobotSpec,
    kinematics: RobotKinematicsModel,
    body_pos: torch.Tensor,
    body_rot_xyzw: torch.Tensor,
    foot_ids: Sequence[int],
    anchor_offsets: torch.Tensor,
    height_axis: int,
) -> torch.Tensor:
    """Return the lowest physical support height for each foot and frame.

    Semantic foot markers remain the IK targets. Configured foot meshes are used
    only for grounding, where the physical sole determines the support height.
    """
    anchor = _marker_positions(body_pos, body_rot_xyzw, foot_ids, anchor_offsets)
    heights = anchor[..., int(height_axis)].clone()
    for slot, name in enumerate(("left_foot", "right_foot")):
        mesh_cfg = robot_spec.foot_grounding_meshes.get(name)
        if mesh_cfg is None:
            continue
        body_id = int(kinematics.body_names.index(mesh_cfg["robot_link"]))
        vertices = torch.as_tensor(_mesh_convex_vertices(mesh_cfg["mesh_path"]), dtype=body_pos.dtype, device=body_pos.device)
        rotation = _quat_xyzw_to_rotmat(body_rot_xyzw[:, body_id])
        world_height = body_pos[:, body_id, int(height_axis), None] + torch.einsum(
            "ti,vi->tv", rotation[:, int(height_axis), :], vertices
        )
        heights[:, slot] = world_height.amin(dim=1)
    return heights


def _smooth_binary(mask: np.ndarray, min_segment: int) -> np.ndarray:
    out = mask.copy()
    for dim in range(out.shape[-1]):
        active = out[:, dim].astype(np.int32)
        start = None
        for idx, value in enumerate(active.tolist() + [0]):
            if value and start is None:
                start = idx
            if start is not None and not value:
                if idx - start < min_segment:
                    active[start:idx] = 0
                start = None
        out[:, dim] = active
    return out


def _hysteresis_binary(values: np.ndarray, enter_threshold: float, exit_threshold: Optional[float]) -> np.ndarray:
    if exit_threshold is None:
        return values >= float(enter_threshold)
    out = np.zeros(values.shape, dtype=bool)
    for dim in range(values.shape[-1]):
        active = False
        for idx, value in enumerate(values[:, dim]):
            if active:
                if float(value) < float(exit_threshold):
                    active = False
            elif float(value) >= float(enter_threshold):
                active = True
            out[idx, dim] = active
    return out


def _fill_short_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    out = mask.copy()
    if max_gap <= 0:
        return out
    for dim in range(out.shape[-1]):
        active = out[:, dim].astype(bool)
        start = None
        for idx, value in enumerate(np.concatenate([active, [True]])):
            if not value and start is None:
                start = idx
            if start is not None and value:
                end = idx
                if start > 0 and end < len(active) and end - start <= int(max_gap):
                    active[start:end] = True
                start = None
        out[:, dim] = active
    return out


def _contact_fade_weights(mask: np.ndarray, fade: int) -> np.ndarray:
    weights = mask.astype(np.float32)
    fade = int(fade)
    if fade <= 0:
        return weights
    for dim in range(mask.shape[-1]):
        active = mask[:, dim].astype(bool)
        start = None
        for idx, flag in enumerate(np.concatenate([active, [False]])):
            if flag and start is None:
                start = idx
            elif start is not None and not flag:
                end = idx
                length = end - start
                if length <= 0:
                    start = None
                    continue
                ramp = min(fade, max(1, length // 2))
                segment = np.ones(length, dtype=np.float32)
                if ramp > 0:
                    segment[:ramp] = np.linspace(1.0 / float(ramp), 1.0, ramp, dtype=np.float32)
                    segment[-ramp:] = np.minimum(
                        segment[-ramp:],
                        np.linspace(1.0, 1.0 / float(ramp), ramp, dtype=np.float32),
                    )
                weights[start:end, dim] = segment
                start = None
    return weights


def _moving_average_1d(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) == 0:
        return values
    kernel = np.ones(int(window), dtype=np.float32) / float(window)
    padded = np.pad(values, ((window // 2, window - 1 - window // 2), (0, 0)), mode="edge")
    smoothed = np.stack([np.convolve(padded[:, dim], kernel, mode="valid") for dim in range(values.shape[-1])], axis=-1)
    return smoothed.astype(np.float32)


def _quat_xyzw_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    if quat.shape[-1] != 4:
        raise ValueError(f"expected quaternion [...,4], got {quat.shape}")
    x, y, z, w = quat.unbind(dim=-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    rot = torch.stack(
        [
            torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], dim=-1),
            torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], dim=-1),
            torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], dim=-1),
        ],
        dim=-2,
    )
    return rot


def _rotmat_to_quat_xyzw(rot: torch.Tensor) -> torch.Tensor:
    trace = rot[..., 0, 0] + rot[..., 1, 1] + rot[..., 2, 2]
    q = torch.zeros(rot.shape[:-2] + (4,), dtype=rot.dtype, device=rot.device)
    cond = trace > 0.0
    s = torch.sqrt(torch.clamp(trace + 1.0, min=1e-8)) * 2.0
    q[..., 3] = 0.25 * s
    q[..., 0] = (rot[..., 2, 1] - rot[..., 1, 2]) / s.clamp(min=1e-8)
    q[..., 1] = (rot[..., 0, 2] - rot[..., 2, 0]) / s.clamp(min=1e-8)
    q[..., 2] = (rot[..., 1, 0] - rot[..., 0, 1]) / s.clamp(min=1e-8)
    if not torch.all(cond):
        alt = torch.zeros_like(q)
        diag = torch.stack([rot[..., 0, 0], rot[..., 1, 1], rot[..., 2, 2]], dim=-1)
        max_idx = torch.argmax(diag, dim=-1)
        for idx in range(3):
            mask = max_idx == idx
            if not torch.any(mask):
                continue
            if idx == 0:
                s = torch.sqrt(torch.clamp(1.0 + rot[..., 0, 0] - rot[..., 1, 1] - rot[..., 2, 2], min=1e-8)) * 2.0
                alt[..., 0] = 0.25 * s
                alt[..., 1] = (rot[..., 0, 1] + rot[..., 1, 0]) / s.clamp(min=1e-8)
                alt[..., 2] = (rot[..., 0, 2] + rot[..., 2, 0]) / s.clamp(min=1e-8)
                alt[..., 3] = (rot[..., 2, 1] - rot[..., 1, 2]) / s.clamp(min=1e-8)
            elif idx == 1:
                s = torch.sqrt(torch.clamp(1.0 + rot[..., 1, 1] - rot[..., 0, 0] - rot[..., 2, 2], min=1e-8)) * 2.0
                alt[..., 0] = (rot[..., 0, 1] + rot[..., 1, 0]) / s.clamp(min=1e-8)
                alt[..., 1] = 0.25 * s
                alt[..., 2] = (rot[..., 1, 2] + rot[..., 2, 1]) / s.clamp(min=1e-8)
                alt[..., 3] = (rot[..., 0, 2] - rot[..., 2, 0]) / s.clamp(min=1e-8)
            else:
                s = torch.sqrt(torch.clamp(1.0 + rot[..., 2, 2] - rot[..., 0, 0] - rot[..., 1, 1], min=1e-8)) * 2.0
                alt[..., 0] = (rot[..., 0, 2] + rot[..., 2, 0]) / s.clamp(min=1e-8)
                alt[..., 1] = (rot[..., 1, 2] + rot[..., 2, 1]) / s.clamp(min=1e-8)
                alt[..., 2] = 0.25 * s
                alt[..., 3] = (rot[..., 1, 0] - rot[..., 0, 1]) / s.clamp(min=1e-8)
            q = torch.where(mask[..., None], alt, q)
    return torch.nn.functional.normalize(q, dim=-1)


def _quat_wxyz_to_xyzw(quat: torch.Tensor) -> torch.Tensor:
    return torch.cat([quat[..., 1:], quat[..., :1]], dim=-1)


def _quat_xyzw_to_wxyz(quat: torch.Tensor) -> torch.Tensor:
    return torch.cat([quat[..., 3:], quat[..., :3]], dim=-1)


def _axis_angle_to_rotmat(axis_angle: torch.Tensor) -> torch.Tensor:
    angle = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)
    axis = axis_angle / angle.clamp(min=1e-8)
    x, y, z = axis.unbind(dim=-1)
    c = torch.cos(angle)[..., 0]
    s = torch.sin(angle)[..., 0]
    one_c = 1.0 - c
    rot = torch.stack(
        [
            torch.stack([c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s], dim=-1),
            torch.stack([y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s], dim=-1),
            torch.stack([z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c], dim=-1),
        ],
        dim=-2,
    )
    eye = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    rot = torch.where((angle[..., 0] < 1e-8)[..., None, None], eye, rot)
    return rot


def _rotmat_from_two_vectors(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    src_norm = src / src.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    dst_norm = dst / dst.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    v = torch.cross(src_norm, dst_norm, dim=-1)
    c = torch.sum(src_norm * dst_norm, dim=-1, keepdim=True)
    s = torch.linalg.norm(v, dim=-1, keepdim=True)
    eye = torch.eye(3, device=src.device, dtype=src.dtype)
    vx = torch.zeros(src.shape[:-1] + (3, 3), device=src.device, dtype=src.dtype)
    vx[..., 0, 1] = -v[..., 2]
    vx[..., 0, 2] = v[..., 1]
    vx[..., 1, 0] = v[..., 2]
    vx[..., 1, 2] = -v[..., 0]
    vx[..., 2, 0] = -v[..., 1]
    vx[..., 2, 1] = v[..., 0]
    rot = eye + vx + vx @ vx * ((1.0 - c) / s.pow(2).clamp(min=1e-8))
    parallel = s[..., 0] < 1e-8
    same = parallel & (c[..., 0] > 0.0)
    opposite = parallel & ~same
    if torch.any(same):
        rot = torch.where(same[..., None, None], eye, rot)
    if torch.any(opposite):
        fallback_axis = torch.zeros_like(src_norm)
        fallback_axis[..., 0] = 1.0
        alt = _axis_angle_to_rotmat(np.pi * fallback_axis)
        rot = torch.where(opposite[..., None, None], alt, rot)
    return rot


def _build_joint_to_dof_lookup(kin: RobotKinematicsModel) -> Dict[int, int]:
    lookup = {}
    for joint_idx, joint in enumerate(kin._joints):  # noqa: SLF001
        if joint.dof_idx >= 0:
            lookup[joint_idx] = joint.dof_idx
    return lookup


def _build_ik_joint_whitelist(kin: RobotKinematicsModel, foot_ids: Sequence[int]) -> set[int]:
    whitelist: set[int] = set()
    for foot_id in foot_ids:
        chain = _chain_from_end_effector(kin._parent_indices, foot_id)  # noqa: SLF001
        whitelist.update(int(j) for j in chain[1:-1])
    return whitelist


def _chain_from_end_effector(parent_indices: torch.Tensor, end_effector: int) -> List[int]:
    chain = [int(end_effector)]
    curr = int(end_effector)
    while curr >= 0:
        curr = int(parent_indices[curr].item())
        if curr >= 0:
            chain.append(curr)
    return list(reversed(chain))


def _joint_local_rot_from_dof(joint, dof_val: torch.Tensor) -> torch.Tensor:
    if joint.dof_dim == 0:
        return torch.eye(3, device=dof_val.device, dtype=dof_val.dtype)
    if joint.dof_dim == 1:
        axis = joint._axis.to(device=dof_val.device, dtype=dof_val.dtype)  # noqa: SLF001
        angle = dof_val[..., :1]
        axis = axis.view(*([1] * (angle.dim() - 1)), 3)
        return _axis_angle_to_rotmat(axis * angle)
    if joint.dof_dim == 3:
        return _axis_angle_to_rotmat(dof_val)
    raise ValueError(f"Unsupported joint dof_dim={joint.dof_dim}")


def _joint_local_rot_to_dof(joint, local_rot: torch.Tensor) -> torch.Tensor:
    aa = rotation_matrix_to_angle_axis(local_rot)
    if joint.dof_dim == 1:
        axis = joint._axis.to(device=local_rot.device, dtype=local_rot.dtype)  # noqa: SLF001
        return (aa * axis).sum(dim=-1, keepdim=True)
    if joint.dof_dim == 3:
        return aa
    return aa.new_zeros((aa.shape[0], 0)) if aa.dim() > 1 else aa.new_zeros((0,))


def _fixed_rotmat_from_quat_xyzw(quat_xyzw: torch.Tensor) -> torch.Tensor:
    quat_xyzw = torch.nn.functional.normalize(quat_xyzw, dim=-1)
    x, y, z, w = quat_xyzw.unbind(dim=-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    rot = torch.stack(
        [
            torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], dim=-1),
            torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], dim=-1),
            torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], dim=-1),
        ],
        dim=-2,
    )
    return rot


def _signed_angle_on_axis(axis_world: torch.Tensor, src_vec: torch.Tensor, dst_vec: torch.Tensor) -> torch.Tensor:
    axis_world = axis_world / axis_world.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    src_proj = src_vec - axis_world * (src_vec * axis_world).sum(dim=-1, keepdim=True)
    dst_proj = dst_vec - axis_world * (dst_vec * axis_world).sum(dim=-1, keepdim=True)
    src_norm = src_proj.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    dst_norm = dst_proj.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    src_unit = src_proj / src_norm
    dst_unit = dst_proj / dst_norm
    cos_angle = torch.clamp((src_unit * dst_unit).sum(dim=-1), -1.0, 1.0)
    sin_angle = (axis_world * torch.cross(src_unit, dst_unit, dim=-1)).sum(dim=-1)
    return torch.atan2(sin_angle, cos_angle)


def _forward_kinematics_dof(
    kin: RobotKinematicsModel,
    root_pos: torch.Tensor,
    root_rot_xyzw: torch.Tensor,
    dof: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    body_pos, body_rot = kin.forward_kinematics(root_pos, root_rot_xyzw, dof)
    return body_pos, body_rot


def _contact_motion_stable_mask(
    foot_pos: np.ndarray,
    height_axis: int,
    speed_threshold_mps: float,
    fps: float,
) -> np.ndarray:
    total = int(foot_pos.shape[0])
    stable = np.ones(foot_pos.shape[:2], dtype=bool)
    if total <= 1:
        return stable

    horizontal_axes = [idx for idx in range(3) if idx != int(height_axis)]
    foot_xy = np.asarray(foot_pos[..., horizontal_axes], dtype=np.float32)
    step = np.linalg.norm(np.diff(foot_xy, axis=0), axis=-1)
    speed = step * float(fps)
    frame_speed = np.zeros_like(stable, dtype=np.float32)
    frame_speed[1:] = np.maximum(frame_speed[1:], speed)
    frame_speed[:-1] = np.maximum(frame_speed[:-1], speed)
    if speed_threshold_mps is not None and float(speed_threshold_mps) > 0.0:
        stable &= frame_speed <= float(speed_threshold_mps)
    return stable


def estimate_contact_mask(contact_prob: np.ndarray, cfg: FootIKConfig) -> np.ndarray:
    support = _hysteresis_binary(
        np.asarray(contact_prob, dtype=np.float32),
        float(cfg.contact_threshold),
        cfg.contact_exit_threshold,
    )
    support = _fill_short_gaps(support.astype(bool), int(cfg.max_contact_gap))
    return _smooth_binary(support.astype(np.int64), int(cfg.min_segment)).astype(bool)


def apply_contact_motion_gate(contact_mask: np.ndarray, cfg: FootIKConfig, foot_pos: Optional[np.ndarray] = None) -> np.ndarray:
    support = np.asarray(contact_mask, dtype=bool)
    if bool(cfg.enable_contact_motion_gate) and foot_pos is not None:
        stable_motion = _contact_motion_stable_mask(
            np.asarray(foot_pos, dtype=np.float32),
            int(cfg.height_axis),
            float(cfg.contact_motion_speed_threshold_mps),
            float(cfg.fps),
        )
        support = support & stable_motion
        support = _smooth_binary(support.astype(np.int64), int(cfg.min_segment)).astype(bool)
    return support


def apply_root_grounding(
    root_pos: np.ndarray,
    support_mask: np.ndarray,
    foot_pos: Optional[np.ndarray] = None,
    height_axis: int = 1,
) -> np.ndarray:
    root_pos = root_pos.copy()
    for t in range(root_pos.shape[0]):
        active = np.where(support_mask[t])[0]
        if len(active) == 0:
            continue
        if foot_pos is not None:
            target_height = float(np.median(foot_pos[t, active, height_axis]))
            root_pos[t, height_axis] -= target_height
        else:
            root_pos[t, height_axis] -= float(np.median(root_pos[:, height_axis]))
    return root_pos


def apply_contact_trajectory_correction(
    root_pos: np.ndarray,
    foot_pos: np.ndarray,
    support_mask: np.ndarray,
    height_axis: int = 1,
    xy_smooth: int = 5,
    xy_gain: float = 1.0,
    xy_clip: float = 0.02,
) -> np.ndarray:
    corrected = root_pos.copy()
    if foot_pos is None or len(foot_pos) == 0:
        return corrected
    horizontal_axes = [idx for idx in range(3) if idx != int(height_axis)]
    active_xy = np.zeros((len(root_pos), 3), dtype=np.float32)
    for t in range(1, len(root_pos)):
        active = support_mask[t] & support_mask[t - 1]
        if not np.any(active):
            continue
        disp = np.median(foot_pos[t, active] - foot_pos[t - 1, active], axis=0)
        disp[height_axis] = 0.0
        active_xy[t:] -= disp
    if xy_smooth > 1:
        active_xy[:, horizontal_axes] = _moving_average_1d(active_xy[:, horizontal_axes], xy_smooth)
    active_xy[:, horizontal_axes] *= float(xy_gain)
    if xy_clip is not None and float(xy_clip) > 0.0:
        active_xy[:, horizontal_axes] = np.clip(active_xy[:, horizontal_axes], -float(xy_clip), float(xy_clip))
    corrected[:, horizontal_axes] += active_xy[:, horizontal_axes]
    return corrected


def apply_root_support_grounding(
    root_pos: np.ndarray,
    support_mask: np.ndarray,
    support_height: np.ndarray,
    height_axis: int = 2,
) -> np.ndarray:
    """Ground the lowest active physical support surface at world height zero."""
    corrected = np.asarray(root_pos, dtype=np.float32).copy()
    active_mask = np.asarray(support_mask, dtype=bool)
    heights = np.asarray(support_height, dtype=np.float32)
    for frame in range(min(len(corrected), len(active_mask), len(heights))):
        active = np.where(active_mask[frame])[0]
        if len(active):
            corrected[frame, int(height_axis)] -= float(np.min(heights[frame, active]))
    return corrected


def apply_root_nonpenetration(
    root_pos: np.ndarray,
    support_height: np.ndarray,
    height_axis: int = 2,
    ground_height: float = 0.0,
    margin: float = 0.003,
    smooth_window: int = 7,
) -> np.ndarray:
    """Lift the root so neither physical foot support surface penetrates the floor.

    Contact remains responsible for foot locking. This constraint applies to
    every frame so a missed contact cannot permit floor penetration.
    """
    corrected = np.asarray(root_pos, dtype=np.float32).copy()
    heights = np.asarray(support_height, dtype=np.float32)
    total = min(len(corrected), len(heights))
    if total == 0:
        return corrected
    if heights.ndim != 2 or heights.shape[1] == 0:
        raise ValueError(f"support_height must be [T, F], got {heights.shape}")

    floor = float(ground_height) + float(margin)
    required = np.maximum(0.0, floor - np.min(heights[:total], axis=1)).astype(np.float32)
    window = max(1, int(smooth_window))
    if window > 1:
        if window % 2 == 0:
            window += 1
        radius = window // 2
        padded = np.pad(required, (radius, radius), mode="edge")
        expanded = np.asarray(
            [np.max(padded[index : index + window]) for index in range(total)],
            dtype=np.float32,
        )
        envelope = _moving_average_1d(expanded[:, None], window)[:, 0]
        correction = np.maximum(required, envelope)
    else:
        correction = required
    corrected[:total, int(height_axis)] += correction
    return corrected


def _contact_segments(mask: np.ndarray, min_length: int = 2) -> List[Tuple[int, int]]:
    segments: List[Tuple[int, int]] = []
    start = None
    for idx, value in enumerate(np.concatenate([mask.astype(bool), [False]])):
        if value and start is None:
            start = idx
        elif start is not None and not value:
            if idx - start >= int(min_length):
                segments.append((start, idx))
            start = None
    return segments


def _fade_segment_weights(length: int, fade: int) -> np.ndarray:
    weights = np.ones(int(length), dtype=np.float32)
    if fade <= 0 or length <= 1:
        return weights
    ramp = min(int(fade), max(1, int(length) // 2))
    values = np.linspace(1.0 / float(ramp), 1.0, ramp, dtype=np.float32)
    weights[:ramp] = np.minimum(weights[:ramp], values)
    weights[-ramp:] = np.minimum(weights[-ramp:], values[::-1])
    return weights


def apply_contact_root_xyz_stabilize(
    root_pos: np.ndarray,
    foot_pos: np.ndarray,
    xy_support_mask: np.ndarray,
    z_support_mask: np.ndarray,
    height_axis: int,
    xy_smooth: int,
    xy_gain: float,
    xy_clip: float,
    z_gain: float,
    z_clip: float,
    fade: int,
) -> np.ndarray:
    total = min(len(root_pos), len(foot_pos), len(xy_support_mask), len(z_support_mask))
    corrected = root_pos.copy()
    if total <= 1:
        return corrected

    foot = np.asarray(foot_pos[:total], dtype=np.float32)
    xy_support = np.asarray(xy_support_mask[:total], dtype=bool)
    z_support = np.asarray(z_support_mask[:total], dtype=bool)
    horizontal_axes = [idx for idx in range(3) if idx != int(height_axis)]
    desired = np.zeros((total, 3), dtype=np.float32)
    weight = np.zeros((total, 1), dtype=np.float32)

    foot_count = min(foot.shape[1], xy_support.shape[1], z_support.shape[1])
    ground = np.zeros((foot_count,), dtype=np.float32)
    for foot_idx in range(foot_count):
        active = z_support[:, foot_idx]
        if np.any(active):
            ground[foot_idx] = float(np.median(foot[active, foot_idx, int(height_axis)]))
        else:
            ground[foot_idx] = float(np.median(foot[:, foot_idx, int(height_axis)]))

    for foot_idx in range(foot_count):
        for start, end in _contact_segments(xy_support[:, foot_idx], min_length=2):
            traj = foot[start:end, foot_idx]
            offset = np.zeros((end - start, 3), dtype=np.float32)
            ref_xy = np.median(traj[:, horizontal_axes], axis=0)
            offset[:, horizontal_axes] = ref_xy[None] - traj[:, horizontal_axes]
            segment_weight = _fade_segment_weights(end - start, int(fade))[:, None]
            desired[start:end] += offset * segment_weight
            weight[start:end] += segment_weight

        for start, end in _contact_segments(z_support[:, foot_idx], min_length=2):
            traj = foot[start:end, foot_idx]
            offset = np.zeros((end - start, 3), dtype=np.float32)
            offset[:, int(height_axis)] = ground[foot_idx] - traj[:, int(height_axis)]
            segment_weight = _fade_segment_weights(end - start, int(fade))[:, None]
            desired[start:end] += offset * segment_weight
            weight[start:end] += segment_weight

    valid = weight[:, 0] > 1e-6
    offset = np.zeros_like(desired)
    offset[valid] = desired[valid] / weight[valid]
    if xy_smooth > 1:
        offset[:, horizontal_axes] = _moving_average_1d(offset[:, horizontal_axes], int(xy_smooth))
        offset[:, int(height_axis) : int(height_axis) + 1] = _moving_average_1d(
            offset[:, int(height_axis) : int(height_axis) + 1],
            int(xy_smooth),
        )

    if xy_clip is not None and float(xy_clip) > 0.0:
        xy_norm = np.linalg.norm(offset[:, horizontal_axes], axis=-1, keepdims=True)
        offset[:, horizontal_axes] *= np.minimum(1.0, float(xy_clip) / np.clip(xy_norm, 1e-8, None))
    if z_clip is not None and float(z_clip) > 0.0:
        offset[:, int(height_axis)] = np.clip(offset[:, int(height_axis)], -float(z_clip), float(z_clip))

    offset[:, horizontal_axes] *= float(xy_gain)
    offset[:, int(height_axis)] *= float(z_gain)
    corrected[:total] += offset
    return corrected.astype(np.float32)


def build_locked_foot_targets(
    foot_pos: np.ndarray,
    support_mask: np.ndarray,
    smooth_window: int = 1,
) -> np.ndarray:
    locked = foot_pos.copy()
    for dim in range(foot_pos.shape[1]):
        active = support_mask[:, dim].astype(bool)
        start = None
        for idx, flag in enumerate(np.concatenate([active, [False]])):
            if flag and start is None:
                start = idx
            elif start is not None and not flag:
                end = idx
                reference = foot_pos[start, dim]
                locked[start:end, dim] = reference
                start = None
    if smooth_window > 1 and len(locked) > 1:
        locked = locked.copy()
        for dim in range(locked.shape[1]):
            locked[:, dim] = _moving_average_1d(locked[:, dim], smooth_window)
    return locked

def _weighted_contact_target(foot_pos: np.ndarray, support_mask: np.ndarray) -> np.ndarray:
    return build_locked_foot_targets(foot_pos, support_mask)


def _smooth_ik_delta(
    source_dof: np.ndarray,
    refined_dof: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    cfg: FootIKConfig,
) -> np.ndarray:
    if not bool(cfg.enable_ik_delta_smoothing):
        return np.clip(refined_dof, lower[None, :], upper[None, :]).astype(np.float32)
    delta = np.asarray(refined_dof - source_dof, dtype=np.float32)
    if int(cfg.ik_delta_smooth_window) > 1 and delta.shape[0] >= 3:
        delta = _moving_average_1d(delta, int(cfg.ik_delta_smooth_window))
    rate_limit = float(cfg.ik_delta_rate_limit)
    if rate_limit > 0.0 and delta.shape[0] > 1:
        limited = delta.copy()
        for idx in range(1, limited.shape[0]):
            step = np.clip(limited[idx] - limited[idx - 1], -rate_limit, rate_limit)
            limited[idx] = limited[idx - 1] + step
        for idx in range(limited.shape[0] - 2, -1, -1):
            step = np.clip(limited[idx] - limited[idx + 1], -rate_limit, rate_limit)
            limited[idx] = limited[idx + 1] + step
        delta = limited
    return np.clip(source_dof + delta, lower[None, :], upper[None, :]).astype(np.float32)


def _ccd_solve_sequence(
    kin: RobotKinematicsModel,
    root_pos: torch.Tensor,
    root_rot_xyzw: torch.Tensor,
    dof_xml: torch.Tensor,
    target_foot_pos: torch.Tensor,
    foot_ids: Sequence[int],
    foot_offsets: torch.Tensor,
    support_mask: torch.Tensor,
    support_weight: Optional[torch.Tensor] = None,
    max_iter: int = 2,
    step_scale: float = 0.12,
    ccd_threshold: float = 1e-3,
    enable_contact_foot_flat: bool = False,
    contact_foot_flat_gain: float = 0.4,
    contact_foot_flat_max_step: float = 0.1,
    contact_foot_flat_local_axis: int = 2,
    height_axis: int = 2,
) -> torch.Tensor:
    refined = dof_xml.clone()
    parent_indices = kin._parent_indices  # noqa: SLF001
    joint_to_dof = _build_joint_to_dof_lookup(kin)
    whitelist = _build_ik_joint_whitelist(kin, foot_ids)
    support_mask = support_mask.bool()
    if support_weight is None:
        support_weight = support_mask.to(dtype=dof_xml.dtype)
    else:
        support_weight = support_weight.to(device=dof_xml.device, dtype=dof_xml.dtype)

    for _ in range(max_iter):
        body_pos, body_rot = _forward_kinematics_dof(kin, root_pos, root_rot_xyzw, refined)
        marker_pos = _marker_positions(body_pos, body_rot, foot_ids, foot_offsets)
        max_err = 0.0
        for foot_slot, foot_id in enumerate(foot_ids):
            active = support_mask[:, foot_slot]
            if not bool(active.any()):
                continue
            target_pos = target_foot_pos[:, foot_slot]
            err = torch.linalg.norm(marker_pos[active, foot_slot] - target_pos[active], dim=-1)
            if err.numel() > 0:
                max_err = max(max_err, float(err.max().item()))

            chain = _chain_from_end_effector(parent_indices, foot_id)
            if len(chain) <= 2:
                continue

            for chain_idx in range(len(chain) - 2, 0, -1):
                joint_id = chain[chain_idx]
                if int(joint_id) not in whitelist:
                    continue
                dof_idx = joint_to_dof.get(joint_id)
                if dof_idx is None:
                    continue

                joint = kin._joints[joint_id]  # noqa: SLF001
                joint_pos = body_pos[:, joint_id]
                end_pos = marker_pos[:, foot_slot]
                src_vec = end_pos - joint_pos
                dst_vec = target_pos - joint_pos
                valid = active & (torch.linalg.norm(src_vec, dim=-1) > 1e-8) & (torch.linalg.norm(dst_vec, dim=-1) > 1e-8)
                if not bool(valid.any()):
                    continue

                parent_id = int(parent_indices[joint_id].item())
                parent_rot = _quat_xyzw_to_rotmat(body_rot[:, parent_id])
                fixed_rot = _fixed_rotmat_from_quat_xyzw(kin._local_rotation[joint_id].to(device=parent_rot.device, dtype=parent_rot.dtype).unsqueeze(0))[0]  # noqa: SLF001
                chain_weight = float(chain_idx) / float(max(len(chain) - 1, 1))
                chain_weight = float(np.clip(chain_weight, 0.15, 1.0))
                if joint.dof_dim == 1:
                    axis = joint._axis.to(device=parent_rot.device, dtype=parent_rot.dtype)  # noqa: SLF001
                    axis_world = torch.einsum("tij,j->ti", parent_rot @ fixed_rot, axis)
                    angle = _signed_angle_on_axis(axis_world, src_vec, dst_vec)
                    step = float(step_scale) * chain_weight
                    weight = support_weight[:, foot_slot]
                    updated = refined[:, dof_idx] + step * angle * weight
                    lower = kin.dof_lower_limits[dof_idx]
                    upper = kin.dof_upper_limits[dof_idx]
                    refined[valid, dof_idx] = torch.clamp(updated[valid], lower, upper)
                elif joint.dof_dim == 3:
                    delta_rot = _rotmat_from_two_vectors(src_vec, dst_vec)
                    current_local = refined[:, dof_idx : dof_idx + 3]
                    current_local_rot = _joint_local_rot_from_dof(joint, current_local)
                    updated_local_rot = delta_rot @ current_local_rot
                    rel_rot = updated_local_rot @ current_local_rot.transpose(-1, -2)
                    rel_aa = rotation_matrix_to_angle_axis(rel_rot)
                    weight = support_weight[:, foot_slot].view(-1, 1)
                    partial_rot = _axis_angle_to_rotmat(rel_aa * chain_weight * weight) @ current_local_rot
                    new_dof = _joint_local_rot_to_dof(joint, partial_rot)
                    lower = kin.dof_lower_limits[dof_idx : dof_idx + 3]
                    upper = kin.dof_upper_limits[dof_idx : dof_idx + 3]
                    refined[valid, dof_idx : dof_idx + 3] = torch.clamp(new_dof[valid], lower, upper)

                body_pos, body_rot = _forward_kinematics_dof(kin, root_pos, root_rot_xyzw, refined)
                marker_pos = _marker_positions(body_pos, body_rot, foot_ids, foot_offsets)

        if enable_contact_foot_flat:
            refined = _apply_contact_foot_flat_correction(
                kin=kin,
                root_pos=root_pos,
                root_rot_xyzw=root_rot_xyzw,
                dof_xml=refined,
                foot_ids=foot_ids,
                support_mask=support_mask,
                support_weight=support_weight,
                gain=float(contact_foot_flat_gain),
                max_step=float(contact_foot_flat_max_step),
                local_normal_axis=int(contact_foot_flat_local_axis),
                height_axis=int(height_axis),
            )

        if max_err < float(ccd_threshold):
            break
    return refined


def _apply_contact_foot_flat_correction(
    kin: RobotKinematicsModel,
    root_pos: torch.Tensor,
    root_rot_xyzw: torch.Tensor,
    dof_xml: torch.Tensor,
    foot_ids: Sequence[int],
    support_mask: torch.Tensor,
    support_weight: torch.Tensor,
    gain: float,
    max_step: float,
    local_normal_axis: int,
    height_axis: int,
) -> torch.Tensor:
    if gain <= 0.0:
        return dof_xml
    local_normal_axis = int(np.clip(local_normal_axis, 0, 2))
    parent_indices = kin._parent_indices  # noqa: SLF001
    joint_to_dof = _build_joint_to_dof_lookup(kin)
    refined = dof_xml.clone()
    world_up = torch.zeros((refined.shape[0], 3), dtype=refined.dtype, device=refined.device)
    world_up[:, int(height_axis)] = 1.0

    for foot_slot, foot_id in enumerate(foot_ids):
        active = support_mask[:, foot_slot].bool()
        if not bool(active.any()):
            continue
        chain = _chain_from_end_effector(parent_indices, foot_id)
        ankle_chain = [
            joint_id
            for joint_id in chain[1:-1]
            if "ankle" in str(kin.body_names[int(joint_id)]).lower()
        ]
        if not ankle_chain:
            ankle_chain = chain[1:-1][-2:]
        if not ankle_chain:
            continue

        for joint_id in reversed(ankle_chain):
            joint_id = int(joint_id)
            dof_idx = joint_to_dof.get(joint_id)
            if dof_idx is None:
                continue
            joint = kin._joints[joint_id]  # noqa: SLF001
            if joint.dof_dim != 1:
                continue

            body_pos, body_rot = _forward_kinematics_dof(kin, root_pos, root_rot_xyzw, refined)
            foot_rot = _quat_xyzw_to_rotmat(body_rot[:, int(foot_id)])
            foot_normal = foot_rot[..., :, local_normal_axis]
            dot = torch.sum(foot_normal[active] * world_up[active], dim=-1)
            target_sign = 1.0 if float(torch.median(dot).item()) >= 0.0 else -1.0
            target_normal = world_up * target_sign

            parent_id = int(parent_indices[joint_id].item())
            parent_rot = _quat_xyzw_to_rotmat(body_rot[:, parent_id])
            fixed_rot = _fixed_rotmat_from_quat_xyzw(
                kin._local_rotation[joint_id].to(device=parent_rot.device, dtype=parent_rot.dtype).unsqueeze(0)
            )[0]  # noqa: SLF001
            axis = joint._axis.to(device=parent_rot.device, dtype=parent_rot.dtype)  # noqa: SLF001
            axis_world = torch.einsum("tij,j->ti", parent_rot @ fixed_rot, axis)
            angle = _signed_angle_on_axis(axis_world, foot_normal, target_normal)
            step = torch.clamp(float(gain) * angle * support_weight[:, foot_slot], -float(max_step), float(max_step))
            lower = kin.dof_lower_limits[dof_idx]
            upper = kin.dof_upper_limits[dof_idx]
            updated = torch.clamp(refined[:, dof_idx] + step, lower, upper)
            refined[active, dof_idx] = updated[active]

    return refined


def refine_robot_motion_with_contact(
    robot_spec: RobotSpec,
    root_pos: np.ndarray,
    root_rot_quat_wxyz: np.ndarray,
    dof: np.ndarray,
    contact_prob: np.ndarray,
    xml_path: str,
    cfg: Optional[FootIKConfig] = None,
) -> Dict[str, np.ndarray]:
    cfg = cfg or FootIKConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # IK works internally in XML dof order. The input/output remains in model order.
    kin = RobotKinematicsModel(xml_path, device=device, model_to_xml_dof=None)
    foot_ids, foot_offsets = _foot_joint_specs(robot_spec, kin)
    foot_offsets_t = torch.as_tensor(foot_offsets, dtype=torch.float32, device=device)
    root_pos_t = torch.from_numpy(root_pos).float().to(device)
    root_rot_t = torch.from_numpy(root_rot_quat_wxyz[..., [1, 2, 3, 0]]).float().to(device)
    dof_t = torch.from_numpy(dof).float().to(device)
    dof_xml_t = model_dof_to_xml_dof(dof_t, robot_spec)
    with torch.no_grad():
        body_pos, body_rot = kin.forward_kinematics(root_pos_t, root_rot_t, dof_xml_t)
        foot_pos = _marker_positions(body_pos, body_rot, foot_ids, foot_offsets_t).detach().cpu().numpy()
        support_height = foot_support_heights(
            robot_spec, kin, body_pos, body_rot, foot_ids, foot_offsets_t, int(cfg.height_axis)
        ).detach().cpu().numpy()

    contact_mask = estimate_contact_mask(contact_prob, cfg)
    support_mask = apply_contact_motion_gate(contact_mask, cfg, foot_pos=foot_pos)
    support_weight = _contact_fade_weights(support_mask, int(cfg.contact_fade)).astype(np.float32)
    height_axis = int(cfg.height_axis)
    if cfg.preserve_root_xy:
        traj_root = root_pos.copy()
    else:
        traj_root = apply_contact_trajectory_correction(
            root_pos,
            foot_pos,
            support_mask,
            height_axis=height_axis,
            xy_smooth=int(cfg.root_xy_smooth),
            xy_gain=float(cfg.root_xy_gain),
            xy_clip=float(cfg.root_xy_clip),
        )
    if robot_spec.foot_grounding_meshes:
        grounded_root = apply_root_support_grounding(traj_root, contact_mask, support_height, height_axis=height_axis)
    else:
        grounded_root = apply_root_grounding(traj_root, contact_mask, foot_pos, height_axis=height_axis)
    if bool(cfg.enable_root_xyz_stabilize):
        grounded_root = apply_contact_root_xyz_stabilize(
            grounded_root,
            foot_pos + (grounded_root - root_pos)[:, None, :],
            xy_support_mask=support_mask,
            z_support_mask=contact_mask,
            height_axis=height_axis,
            xy_smooth=int(cfg.root_xy_smooth),
            xy_gain=float(cfg.root_xy_gain),
            xy_clip=float(cfg.root_xyz_xy_clip),
            z_gain=float(cfg.root_xyz_z_gain),
            z_clip=float(cfg.root_xyz_z_clip),
            fade=int(cfg.root_xyz_fade),
        )
    root_delta = grounded_root - root_pos
    post_root_foot_pos = foot_pos + root_delta[:, None, :]
    locked_target_foot = _weighted_contact_target(post_root_foot_pos, support_mask)

    if not cfg.enable_ik:
        return {
            "root_pos": grounded_root.astype(np.float32),
            "root_rot_quat_wxyz": root_rot_quat_wxyz.astype(np.float32),
            "dof": dof.astype(np.float32),
            "support_mask": support_mask.astype(np.bool_),
            "contact_mask": contact_mask.astype(np.bool_),
            "support_weight": support_weight.astype(np.float32),
        }

    grounded_root_t = torch.from_numpy(grounded_root).float().to(device)
    target_foot = build_locked_foot_targets(locked_target_foot, support_mask, smooth_window=int(cfg.foot_target_smooth))
    target_foot_t = torch.from_numpy(target_foot).float().to(device)
    refined_xml = dof_xml_t.clone()
    root_rot_t = root_rot_t.detach()

    refined_xml = _ccd_solve_sequence(
        kin=kin,
        root_pos=grounded_root_t,
        root_rot_xyzw=root_rot_t,
        dof_xml=refined_xml,
        target_foot_pos=target_foot_t,
        foot_ids=foot_ids,
        foot_offsets=foot_offsets_t,
        support_mask=torch.from_numpy(support_mask).to(device=device),
        support_weight=torch.from_numpy(support_weight).to(device=device),
        max_iter=int(cfg.ik_steps),
        step_scale=float(cfg.ccd_step_scale),
        ccd_threshold=float(cfg.ccd_threshold),
        enable_contact_foot_flat=bool(cfg.enable_contact_foot_flat),
        contact_foot_flat_gain=float(cfg.contact_foot_flat_gain),
        contact_foot_flat_max_step=float(cfg.contact_foot_flat_max_step),
        contact_foot_flat_local_axis=int(cfg.contact_foot_flat_local_axis),
        height_axis=int(cfg.height_axis),
    )

    refined_xml = refined_xml.detach().cpu().numpy().astype(np.float32)
    lower_xml = kin.dof_lower_limits.cpu().numpy()
    upper_xml = kin.dof_upper_limits.cpu().numpy()
    refined_xml = _smooth_ik_delta(
        dof_xml_t.detach().cpu().numpy().astype(np.float32),
        refined_xml,
        lower_xml,
        upper_xml,
        cfg,
    )
    if robot_spec.model_to_xml_dof is not None:
        refined = refined_xml[..., robot_spec.model_to_xml_dof]
    else:
        refined = refined_xml
    return {
        "root_pos": grounded_root.astype(np.float32),
        "root_rot_quat_wxyz": root_rot_quat_wxyz.astype(np.float32),
        "dof": refined,
        "support_mask": support_mask.astype(np.bool_),
        "contact_mask": contact_mask.astype(np.bool_),
        "support_weight": support_weight.astype(np.float32),
    }
