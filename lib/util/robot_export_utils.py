import json
import os
from typing import Dict, Optional

import numpy as np
import torch

from lib.robot.robot_kinematics import RobotKinematicsModel


def finite_difference(values: np.ndarray, fps: int) -> np.ndarray:
    vel = np.zeros_like(values, dtype=np.float32)
    if values.shape[0] <= 1:
        return vel
    vel[1:] = (values[1:] - values[:-1]) * float(fps)
    vel[0] = vel[1]
    return vel


def quat_mul_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = np.split(a, 4, axis=-1)
    bx, by, bz, bw = np.split(b, 4, axis=-1)
    return np.concatenate(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )


def quat_conjugate_xyzw(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., :3] *= -1.0
    return out


def quat_to_angvel(body_quat_xyzw: np.ndarray, fps: int) -> np.ndarray:
    angvel = np.zeros(body_quat_xyzw.shape[:-1] + (3,), dtype=np.float32)
    if body_quat_xyzw.shape[0] <= 1:
        return angvel
    dq = quat_mul_xyzw(body_quat_xyzw[1:], quat_conjugate_xyzw(body_quat_xyzw[:-1]))
    xyz = dq[..., :3]
    w = np.clip(dq[..., 3:4], -1.0, 1.0)
    norm_xyz = np.linalg.norm(xyz, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(norm_xyz, w)
    axis = np.zeros_like(xyz)
    valid = norm_xyz[..., 0] > 1e-8
    axis[valid] = xyz[valid] / norm_xyz[valid]
    angvel[1:] = axis * angle * float(fps)
    angvel[0] = angvel[1]
    return angvel


def export_robot_predictions(
    sequence_dir: str,
    prediction: Dict[str, np.ndarray],
    xml_path: str,
    robot_name: str,
    model_to_xml_dof: Optional[list] = None,
    neutral_dof: Optional[list] = None,
) -> Dict[str, str]:
    os.makedirs(sequence_dir, exist_ok=True)
    fps = int(np.asarray(prediction["fps"]).reshape(-1)[0])
    root_trans = prediction["root_trans"].astype(np.float32)
    root_rot_quat = prediction["root_rot_quat"].astype(np.float32)
    root_rot_6d = prediction["root_rot_6d"].astype(np.float32)
    dof = prediction["dof"].astype(np.float32)

    raw_path = os.path.join(sequence_dir, f"{robot_name}_raw_pred.npz")
    np.savez_compressed(
        raw_path,
        fps=np.array([fps], dtype=np.int64),
        robot_name=np.array([robot_name]),
        xml_path=np.array([xml_path]),
        root_trans=root_trans,
        root_rot_quat=root_rot_quat,
        root_rot_6d=root_rot_6d,
        dof=dof,
    )

    kinematics = RobotKinematicsModel(xml_path, device="cpu", model_to_xml_dof=model_to_xml_dof, neutral_dof=neutral_dof)
    with torch.no_grad():
        body_pos_t, body_quat_xyzw_t = kinematics.forward_kinematics(
            torch.from_numpy(root_trans),
            torch.from_numpy(root_rot_quat[:, [1, 2, 3, 0]]),
            torch.from_numpy(dof),
        )
    body_pos = body_pos_t.cpu().numpy().astype(np.float32)
    body_quat_xyzw = body_quat_xyzw_t.cpu().numpy().astype(np.float32)
    body_quat_wxyz = body_quat_xyzw[..., [3, 0, 1, 2]]
    joint_vel = finite_difference(dof, fps)
    body_lin_vel = finite_difference(body_pos, fps)
    body_ang_vel = quat_to_angvel(body_quat_xyzw, fps)

    body_path = os.path.join(sequence_dir, f"{robot_name}_body_pred.npz")
    np.savez_compressed(
        body_path,
        fps=np.array([fps], dtype=np.int64),
        robot_name=np.array([robot_name]),
        joint_pos=dof,
        joint_vel=joint_vel,
        body_pos_w=body_pos,
        body_quat_w=body_quat_wxyz.astype(np.float32),
        body_lin_vel_w=body_lin_vel.astype(np.float32),
        body_ang_vel_w=body_ang_vel.astype(np.float32),
        body_names=np.asarray(kinematics.body_names),
    )

    meta_path = os.path.join(sequence_dir, f"{robot_name}_export_meta.json")
    with open(meta_path, "w", encoding="utf-8") as file_obj:
        json.dump(
            {
                "fps": fps,
                "frames": int(dof.shape[0]),
                "robot_name": robot_name,
                "robot_dof": int(dof.shape[1]),
                "xml_path": xml_path,
                "body_names": list(kinematics.body_names),
            },
            file_obj,
            ensure_ascii=False,
            indent=2,
        )
    return {
        f"{robot_name}_raw_pred": raw_path,
        f"{robot_name}_body_pred": body_path,
        f"{robot_name}_export_meta": meta_path,
    }
