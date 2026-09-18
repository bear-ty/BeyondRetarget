#!/usr/bin/env python3
"""ZMQ publisher for GR00T WholeBodyControl streaming motion protocol.

The GR00T deployment subscriber expects a single ZMQ message:

    [topic bytes][1280-byte JSON header][packed binary payload]

This module publishes Protocol v1 joint-based G1 motion frames from RGB2Robo
retargeted actions.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HEADER_SIZE = 1280
G1_DOF = 29

# GR00T's IsaacLab order for G1 29-DOF joints. RGB2Robo's live G1 output is
# expected to already be in this NMR/IsaacLab-style interleaved order.
G1_ISAACLAB_JOINT_ORDER = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)

# MuJoCo/XML/URDF order -> GR00T IsaacLab order.
G1_MUJOCO_TO_ISAACLAB = np.asarray(
    [
        0,
        6,
        12,
        1,
        7,
        13,
        2,
        8,
        14,
        3,
        9,
        15,
        22,
        4,
        10,
        16,
        23,
        5,
        11,
        17,
        24,
        18,
        25,
        19,
        26,
        20,
        27,
        21,
        28,
    ],
    dtype=np.int64,
)


def _build_header(fields: Sequence[Dict[str, object]], *, version: int, count: int) -> bytes:
    header = {
        "v": int(version),
        "endian": "le",
        "count": int(count),
        "fields": list(fields),
    }
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(header_json) > HEADER_SIZE:
        raise ValueError(f"GR00T ZMQ header too large: {len(header_json)} > {HEADER_SIZE}")
    return header_json.ljust(HEADER_SIZE, b"\x00")


def _dtype_name(array: np.ndarray) -> str:
    dtype = np.dtype(array.dtype)
    if dtype == np.dtype(np.float32):
        return "f32"
    if dtype == np.dtype(np.float64):
        return "f64"
    if dtype == np.dtype(np.int32):
        return "i32"
    if dtype == np.dtype(np.int64):
        return "i64"
    if dtype == np.dtype(np.uint8):
        return "u8"
    if dtype == np.dtype(np.bool_):
        return "bool"
    raise ValueError(f"unsupported GR00T ZMQ dtype: {dtype}")


def pack_groot_pose_message(
    fields_by_name: Dict[str, np.ndarray],
    *,
    topic: str = "pose",
    version: int = 1,
    count: int = 1,
) -> bytes:
    """Pack fields into GR00T's single-part ZMQ wire format."""

    fields = []
    payload_parts = []
    for name, value in fields_by_name.items():
        array = np.asarray(value)
        if array.dtype.byteorder == ">":
            array = array.astype(array.dtype.newbyteorder("<"))
        array = np.ascontiguousarray(array)
        fields.append({"name": str(name), "dtype": _dtype_name(array), "shape": list(array.shape)})
        payload_parts.append(array.tobytes(order="C"))

    header = _build_header(fields, version=version, count=count)
    return topic.encode("utf-8") + header + b"".join(payload_parts)


def g1_rot6d_to_quat_wxyz(rot6d: np.ndarray) -> np.ndarray:
    """Decode RGB2Robo G1 root 6D rotations into normalized wxyz quaternions."""

    arr = np.asarray(rot6d, dtype=np.float32)
    if arr.shape[-1] != 6:
        raise ValueError(f"expected root_rot_6d last dim 6, got {arr.shape}")
    flat = arr.reshape(-1, 6)
    row1 = flat[:, 0:3]
    row1 = row1 / np.clip(np.linalg.norm(row1, axis=1, keepdims=True), 1e-6, None)
    row2_raw = flat[:, 3:6]
    row2 = row2_raw - np.sum(row1 * row2_raw, axis=1, keepdims=True) * row1
    row2 = row2 / np.clip(np.linalg.norm(row2, axis=1, keepdims=True), 1e-6, None)
    row3 = np.cross(row1, row2)
    rotmat = np.stack([row1, row2, row3], axis=1).reshape(*arr.shape[:-1], 3, 3)
    return rotation_matrix_to_quat_wxyz(rotmat)


def rotation_matrix_to_quat_wxyz(rotmat: np.ndarray) -> np.ndarray:
    """Convert rotation matrices to wxyz quaternions."""

    mat = np.asarray(rotmat, dtype=np.float32)
    if mat.shape[-2:] != (3, 3):
        raise ValueError(f"expected rotation matrix shape (..., 3, 3), got {mat.shape}")
    m00 = mat[..., 0, 0]
    m01 = mat[..., 0, 1]
    m02 = mat[..., 0, 2]
    m10 = mat[..., 1, 0]
    m11 = mat[..., 1, 1]
    m12 = mat[..., 1, 2]
    m20 = mat[..., 2, 0]
    m21 = mat[..., 2, 1]
    m22 = mat[..., 2, 2]

    qw = 0.5 * np.sqrt(np.maximum(0.0, 1.0 + m00 + m11 + m22))
    qx = 0.5 * np.sqrt(np.maximum(0.0, 1.0 + m00 - m11 - m22))
    qy = 0.5 * np.sqrt(np.maximum(0.0, 1.0 - m00 + m11 - m22))
    qz = 0.5 * np.sqrt(np.maximum(0.0, 1.0 - m00 - m11 + m22))

    qx = np.copysign(qx, m21 - m12)
    qy = np.copysign(qy, m02 - m20)
    qz = np.copysign(qz, m10 - m01)
    quat = np.stack([qw, qx, qy, qz], axis=-1).astype(np.float32)
    return normalize_quat_wxyz(quat)


def normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    arr = np.asarray(quat, dtype=np.float32)
    if arr.shape[-1] != 4:
        raise ValueError(f"expected quaternion last dim 4, got {arr.shape}")
    norm = np.linalg.norm(arr, axis=-1, keepdims=True)
    out = arr / np.clip(norm, 1e-6, None)
    return out.astype(np.float32)


def reorder_g1_joint_pos(joint_pos: np.ndarray, source_joint_order: str = "nmr") -> np.ndarray:
    """Return joint positions in GR00T IsaacLab order."""

    arr = np.asarray(joint_pos, dtype=np.float32)
    if arr.shape[-1] != G1_DOF:
        raise ValueError(f"expected {G1_DOF} G1 joints, got shape {arr.shape}")
    order = source_joint_order.lower()
    if order in {"nmr", "isaaclab", "groot"}:
        return np.ascontiguousarray(arr, dtype=np.float32)
    if order in {"xml", "mujoco", "urdf"}:
        return np.ascontiguousarray(arr[..., G1_MUJOCO_TO_ISAACLAB], dtype=np.float32)
    raise ValueError(f"unsupported G1 joint order: {source_joint_order}")


def _as_frame_batch(array: np.ndarray, width: int, name: str) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] != width:
        raise ValueError(f"expected {name} shape [N, {width}] or [{width}], got {arr.shape}")
    return np.ascontiguousarray(arr, dtype=np.float32)


def _make_frame_index(
    source_frame_ids: Optional[Sequence[int]],
    *,
    count: int,
    next_frame_index: int,
    frame_index_source: str,
) -> np.ndarray:
    if frame_index_source == "source" and source_frame_ids is not None:
        frame_index = np.asarray(source_frame_ids, dtype=np.int64).reshape(-1)
        if frame_index.size != count:
            raise ValueError(f"expected {count} source frame ids, got {frame_index.size}")
        return frame_index
    return np.arange(next_frame_index, next_frame_index + count, dtype=np.int64)


def _estimate_joint_vel(
    joint_pos: np.ndarray,
    frame_index: np.ndarray,
    *,
    fps: float,
    previous_joint_pos: Optional[np.ndarray],
    previous_frame_index: Optional[int],
) -> np.ndarray:
    vel = np.zeros_like(joint_pos, dtype=np.float32)
    if joint_pos.shape[0] == 0:
        return vel
    fps = max(float(fps), 1e-6)

    if previous_joint_pos is not None and previous_frame_index is not None:
        frame_delta = max(int(frame_index[0]) - int(previous_frame_index), 1)
        vel[0] = (joint_pos[0] - previous_joint_pos) * (fps / float(frame_delta))
    elif joint_pos.shape[0] > 1:
        frame_delta = max(int(frame_index[1]) - int(frame_index[0]), 1)
        vel[0] = (joint_pos[1] - joint_pos[0]) * (fps / float(frame_delta))

    for idx in range(1, joint_pos.shape[0]):
        frame_delta = max(int(frame_index[idx]) - int(frame_index[idx - 1]), 1)
        vel[idx] = (joint_pos[idx] - joint_pos[idx - 1]) * (fps / float(frame_delta))
    return vel.astype(np.float32)


def _stabilize_quat_sequence(quat: np.ndarray, previous_quat: Optional[np.ndarray]) -> np.ndarray:
    out = normalize_quat_wxyz(quat).reshape(-1, 4).copy()
    ref = None if previous_quat is None else np.asarray(previous_quat, dtype=np.float32).reshape(4)
    for idx in range(out.shape[0]):
        if ref is not None and float(np.dot(ref, out[idx])) < 0.0:
            out[idx] *= -1.0
        ref = out[idx]
    return out.reshape(quat.shape).astype(np.float32)


def build_groot_v1_fields(
    *,
    joint_pos: np.ndarray,
    root_rot_6d: Optional[np.ndarray] = None,
    root_quat_wxyz: Optional[np.ndarray] = None,
    source_frame_ids: Optional[Sequence[int]] = None,
    fps: float = 30.0,
    source_joint_order: str = "nmr",
    frame_index_source: str = "sequential",
    next_frame_index: int = 0,
    previous_joint_pos: Optional[np.ndarray] = None,
    previous_frame_index: Optional[int] = None,
    previous_root_quat: Optional[np.ndarray] = None,
    catch_up: bool = True,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Build Protocol v1 fields and return them with the generated frame index."""

    joint_batch = _as_frame_batch(joint_pos, G1_DOF, "joint_pos")
    joint_batch = reorder_g1_joint_pos(joint_batch, source_joint_order)
    count = int(joint_batch.shape[0])
    frame_index = _make_frame_index(
        source_frame_ids,
        count=count,
        next_frame_index=next_frame_index,
        frame_index_source=frame_index_source,
    )
    joint_vel = _estimate_joint_vel(
        joint_batch,
        frame_index,
        fps=fps,
        previous_joint_pos=previous_joint_pos,
        previous_frame_index=previous_frame_index,
    )

    if root_quat_wxyz is not None:
        body_quat = _as_frame_batch(root_quat_wxyz, 4, "root_quat_wxyz")
    elif root_rot_6d is not None:
        root_rot_batch = _as_frame_batch(root_rot_6d, 6, "root_rot_6d")
        if root_rot_batch.shape[0] != count:
            raise ValueError(f"root_rot_6d frame count {root_rot_batch.shape[0]} != joint frame count {count}")
        body_quat = g1_rot6d_to_quat_wxyz(root_rot_batch)
    else:
        body_quat = np.zeros((count, 4), dtype=np.float32)
        body_quat[:, 0] = 1.0
    if body_quat.shape[0] != count:
        raise ValueError(f"body_quat frame count {body_quat.shape[0]} != joint frame count {count}")
    body_quat = _stabilize_quat_sequence(body_quat.astype(np.float32), previous_root_quat)

    fields = {
        "joint_pos": joint_batch.astype(np.float32),
        "joint_vel": joint_vel.astype(np.float32),
        "body_quat": body_quat.astype(np.float32),
        "frame_index": frame_index.astype(np.int64),
        "timestamp_monotonic": np.asarray([time.monotonic()], dtype=np.float64),
        "catch_up": np.asarray([bool(catch_up)], dtype=np.bool_),
    }
    return fields, frame_index


@dataclass
class GrootPublisherStats:
    enabled: bool = False
    endpoint: Optional[str] = None
    topic: str = "pose"
    protocol_version: int = 1
    source_joint_order: str = "nmr"
    frame_index_source: str = "sequential"
    single_frame: bool = False
    window_size: int = 50
    warmup_full_window: bool = True
    window_alignment: str = "lead"
    payload_frames_per_message: int = 50
    publish_fps: Optional[float] = None
    messages: int = 0
    frames: int = 0
    accepted_frames: int = 0
    payload_frames_sent: int = 0
    bytes_sent: int = 0
    dropped_messages: int = 0
    first_frame_index: Optional[int] = None
    last_frame_index: Optional[int] = None
    error: Optional[str] = None

    def as_dict(self) -> Dict[str, object]:
        return {
            "enabled": self.enabled,
            "endpoint": self.endpoint,
            "topic": self.topic,
            "protocol_version": self.protocol_version,
            "source_joint_order": self.source_joint_order,
            "frame_index_source": self.frame_index_source,
            "single_frame": self.single_frame,
            "window_size": self.window_size,
            "warmup_full_window": self.warmup_full_window,
            "window_alignment": self.window_alignment,
            "payload_frames_per_message": self.payload_frames_per_message,
            "publish_fps": self.publish_fps,
            "messages": self.messages,
            "frames": self.frames,
            "accepted_frames": self.accepted_frames,
            "payload_frames_sent": self.payload_frames_sent,
            "bytes_sent": self.bytes_sent,
            "dropped_messages": self.dropped_messages,
            "first_frame_index": self.first_frame_index,
            "last_frame_index": self.last_frame_index,
            "error": self.error,
        }


class GrootZMQPublisher:
    """Publish RGB2Robo retargeted G1 actions in GR00T ZMQ Protocol v1."""

    def __init__(
        self,
        *,
        bind_host: str = "*",
        port: int = 5556,
        topic: str = "pose",
        fps: float = 30.0,
        source_joint_order: str = "nmr",
        frame_index_source: str = "sequential",
        catch_up: bool = True,
        window_size: int = 50,
        warmup_full_window: bool = True,
        window_alignment: str = "lead",
        publish_fps: Optional[float] = None,
        send_hwm: int = 10,
        linger_ms: int = 0,
    ) -> None:
        try:
            import zmq
        except ImportError as exc:
            raise RuntimeError("pyzmq is required for --enable_groot_zmq") from exc

        self._zmq = zmq
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.setsockopt(zmq.LINGER, int(linger_ms))
        if int(send_hwm) > 0:
            self._socket.setsockopt(zmq.SNDHWM, int(send_hwm))
        self.endpoint = f"tcp://{bind_host}:{int(port)}"
        self._socket.bind(self.endpoint)

        self.topic = str(topic)
        self.fps = float(fps)
        self.source_joint_order = str(source_joint_order)
        self.frame_index_source = str(frame_index_source)
        self.catch_up = bool(catch_up)
        self.window_size = max(1, int(window_size))
        self.warmup_full_window = bool(warmup_full_window)
        self.window_alignment = str(window_alignment)
        self.publish_fps = max(float(publish_fps) if publish_fps is not None else self.fps, 1e-6)
        self.resample_delay_sec = 1.0 / max(self.fps, 1e-6)
        self.max_source_hold_sec = max(0.25, 4.0 / max(self.fps, 1e-6))
        self._next_frame_index = 0
        self._previous_joint_pos: Optional[np.ndarray] = None
        self._previous_frame_index: Optional[int] = None
        self._previous_root_quat: Optional[np.ndarray] = None
        self._next_output_frame_index = 0
        self._last_output_joint_pos: Optional[np.ndarray] = None
        self._source_samples: List[Tuple[float, np.ndarray, np.ndarray]] = []
        self._source_sample_limit = 8
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._sender_thread = threading.Thread(target=self._send_loop, name="groot-zmq-publisher", daemon=True)
        self.stats = GrootPublisherStats(
            enabled=True,
            endpoint=self.endpoint,
            topic=self.topic,
            source_joint_order=self.source_joint_order,
            frame_index_source=self.frame_index_source,
            single_frame=self.window_size == 1,
            window_size=self.window_size,
            warmup_full_window=self.warmup_full_window,
            window_alignment=self.window_alignment,
            payload_frames_per_message=self.window_size,
            publish_fps=self.publish_fps,
        )
        self._sender_thread.start()

    def close(self) -> None:
        self._stop_event.set()
        if self._sender_thread.is_alive():
            self._sender_thread.join(timeout=1.0)
        if self._socket is not None:
            self._socket.close(0)
            self._socket = None

    def __enter__(self) -> "GrootZMQPublisher":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def publish(
        self,
        *,
        joint_pos: np.ndarray,
        root_rot_6d: Optional[np.ndarray] = None,
        root_quat_wxyz: Optional[np.ndarray] = None,
        source_frame_ids: Optional[Sequence[int]] = None,
    ) -> bool:
        fields, frame_index = build_groot_v1_fields(
            joint_pos=joint_pos,
            root_rot_6d=root_rot_6d,
            root_quat_wxyz=root_quat_wxyz,
            source_frame_ids=source_frame_ids,
            fps=self.fps,
            source_joint_order=self.source_joint_order,
            frame_index_source=self.frame_index_source,
            next_frame_index=self._next_frame_index,
            previous_joint_pos=self._previous_joint_pos,
            previous_frame_index=self._previous_frame_index,
            previous_root_quat=self._previous_root_quat,
            catch_up=self.catch_up,
        )
        new_count = int(fields["joint_pos"].shape[0])

        now_ts = time.perf_counter()
        with self._lock:
            self._previous_joint_pos = fields["joint_pos"][-1].copy()
            self._previous_frame_index = int(frame_index[-1])
            self._previous_root_quat = fields["body_quat"][-1].copy()
            if self.frame_index_source != "source":
                self._next_frame_index += new_count
            elif source_frame_ids is None:
                self._next_frame_index += new_count
            self.stats.accepted_frames += new_count

            for idx in range(new_count):
                joint_pos_i = np.asarray(fields["joint_pos"][idx], dtype=np.float32).reshape(G1_DOF).copy()
                body_quat_i = normalize_quat_wxyz(np.asarray(fields["body_quat"][idx], dtype=np.float32).reshape(4))
                if self._source_samples:
                    prev_quat = self._source_samples[-1][2]
                    if float(np.dot(prev_quat, body_quat_i)) < 0.0:
                        body_quat_i = -body_quat_i
                sample_time = now_ts + float(idx) / max(self.fps, 1e-6)
                self._source_samples.append((sample_time, joint_pos_i, body_quat_i.copy()))
                if len(self._source_samples) > self._source_sample_limit:
                    del self._source_samples[: len(self._source_samples) - self._source_sample_limit]
        return True

    def _send_loop(self) -> None:
        period = 1.0 / max(self.publish_fps, 1e-6)
        next_tick = time.perf_counter()
        try:
            while not self._stop_event.is_set():
                next_tick += period
                wait = next_tick - time.perf_counter()
                if wait > 0.0:
                    self._stop_event.wait(wait)
                    if self._stop_event.is_set():
                        break
                now_ts = time.perf_counter()
                if next_tick < now_ts - period:
                    next_tick = now_ts
                self._send_resampled_frame(now_ts)
        except BaseException as exc:  # noqa: BLE001
            with self._lock:
                self.stats.error = repr(exc)
            self._stop_event.set()

    def _send_resampled_frame(self, now_ts: float) -> bool:
        return self._append_output_window(now_ts - self.resample_delay_sec, now_ts)

    def _interpolate_source_pose(self, target_time: float, now_ts: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        with self._lock:
            samples: List[Tuple[float, np.ndarray, np.ndarray]] = list(self._source_samples)
        if not samples:
            return None
        latest_time, latest_joint, latest_quat = samples[-1]
        if now_ts - latest_time > self.max_source_hold_sec:
            return None
        if len(samples) == 1:
            return latest_joint.copy(), latest_quat.copy()

        prev_sample = samples[0]
        next_sample = samples[-1]
        for idx in range(1, len(samples)):
            if samples[idx][0] >= target_time:
                prev_sample = samples[idx - 1]
                next_sample = samples[idx]
                break

        t0, joint0, quat0 = prev_sample
        t1, joint1, quat1 = next_sample
        if target_time <= samples[0][0]:
            return samples[0][1].copy(), samples[0][2].copy()
        if target_time >= samples[-1][0]:
            return latest_joint.copy(), latest_quat.copy()

        alpha = float(np.clip((target_time - t0) / max(t1 - t0, 1e-9), 0.0, 1.0))
        joint = ((1.0 - alpha) * joint0 + alpha * joint1).astype(np.float32)
        quat = normalize_quat_wxyz(((1.0 - alpha) * quat0 + alpha * quat1).astype(np.float32))
        return joint, quat

    def _append_output_window(self, base_target_time: float, now_ts: float) -> bool:
        joint_window: List[np.ndarray] = []
        quat_window: List[np.ndarray] = []
        prev_quat: Optional[np.ndarray] = None

        for idx in range(self.window_size):
            target_time = base_target_time + float(idx) / max(self.publish_fps, 1e-6)
            pose = self._interpolate_source_pose(target_time, now_ts)
            if pose is None:
                return False
            joint_pos, body_quat = pose
            joint_pos = np.asarray(joint_pos, dtype=np.float32).reshape(G1_DOF)
            body_quat = normalize_quat_wxyz(np.asarray(body_quat, dtype=np.float32).reshape(4))
            if prev_quat is not None and float(np.dot(prev_quat, body_quat)) < 0.0:
                body_quat = -body_quat
            joint_window.append(joint_pos.copy())
            quat_window.append(body_quat.copy())
            prev_quat = body_quat

        with self._lock:
            frame_index_start = int(self._next_output_frame_index)
            self._next_output_frame_index += 1

            joint_pos_batch = np.stack(joint_window, axis=0).astype(np.float32)
            body_quat_batch = np.stack(quat_window, axis=0).astype(np.float32)
            joint_vel_batch = np.zeros_like(joint_pos_batch, dtype=np.float32)
            if self._last_output_joint_pos is not None:
                joint_vel_batch[0] = (
                    (joint_pos_batch[0] - self._last_output_joint_pos) * self.publish_fps
                ).astype(np.float32)
            else:
                joint_vel_batch[0] = np.zeros(G1_DOF, dtype=np.float32)
            if self.window_size > 1:
                joint_vel_batch[1:] = ((joint_pos_batch[1:] - joint_pos_batch[:-1]) * self.publish_fps).astype(np.float32)
            self._last_output_joint_pos = joint_pos_batch[0].copy()

            frame_index_batch = np.arange(
                frame_index_start,
                frame_index_start + self.window_size,
                dtype=np.int64,
            )

            pose_send_monotonic = time.monotonic()
            send_fields = {
                "joint_pos": joint_pos_batch,
                "joint_vel": joint_vel_batch,
                "body_quat": body_quat_batch,
                "frame_index": frame_index_batch,
                "timestamp_monotonic": np.asarray([pose_send_monotonic], dtype=np.float64),
                "pose_send_monotonic": np.asarray([pose_send_monotonic], dtype=np.float64),
                "catch_up": np.asarray([bool(self.catch_up)], dtype=np.bool_),
            }
        send_count = int(send_fields["joint_pos"].shape[0])
        message = pack_groot_pose_message(send_fields, topic=self.topic, version=1, count=send_count)
        try:
            self._socket.send(message, flags=self._zmq.NOBLOCK)
        except self._zmq.Again:
            with self._lock:
                self.stats.dropped_messages += 1
            return False

        with self._lock:
            self.stats.messages += 1
            self.stats.frames += 1
            self.stats.payload_frames_sent += send_count
            self.stats.bytes_sent += len(message)
            if self.stats.first_frame_index is None:
                self.stats.first_frame_index = int(send_fields["frame_index"][0])
            self.stats.last_frame_index = int(send_fields["frame_index"][-1])
        return True

    def meta(self) -> Dict[str, object]:
        with self._lock:
            return self.stats.as_dict()


def replay_npz(
    *,
    npz_path: Path,
    publisher: GrootZMQPublisher,
    batch_size: int = 1,
    replay_fps: Optional[float] = None,
    start_index: int = 0,
    max_frames: Optional[int] = None,
    loop: bool = False,
) -> None:
    data = np.load(npz_path, allow_pickle=False)
    dof = np.asarray(data["dof"], dtype=np.float32)
    if dof.ndim != 2 or dof.shape[1] != G1_DOF:
        raise ValueError(f"{npz_path} has invalid dof shape {dof.shape}")
    root_quat = np.asarray(data["root_rot_quat"], dtype=np.float32) if "root_rot_quat" in data.files else None
    root_rot6d = np.asarray(data["root_rot_6d"], dtype=np.float32) if "root_rot_6d" in data.files else None
    source_ids = np.asarray(data["source_frame_ids"], dtype=np.int64) if "source_frame_ids" in data.files else None
    if replay_fps is None:
        replay_fps = float(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data.files else publisher.fps

    start = max(0, int(start_index))
    end = dof.shape[0] if max_frames is None else min(dof.shape[0], start + max(0, int(max_frames)))
    batch_size = max(1, int(batch_size))
    frame_delay = 1.0 / max(float(replay_fps), 1e-6)

    while True:
        idx = start
        while idx < end:
            next_idx = min(end, idx + batch_size)
            publisher.publish(
                joint_pos=dof[idx:next_idx],
                root_rot_6d=None if root_rot6d is None else root_rot6d[idx:next_idx],
                root_quat_wxyz=None if root_quat is None else root_quat[idx:next_idx],
                source_frame_ids=None if source_ids is None else source_ids[idx:next_idx],
            )
            time.sleep(frame_delay * float(next_idx - idx))
            idx = next_idx
        if not loop:
            break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish RGB2Robo action NPZ to GR00T ZMQ Protocol v1.")
    parser.add_argument("--npz", required=True, type=Path, help="Path to g1_action_stream.npz")
    parser.add_argument("--bind_host", default="*", help="ZMQ bind host, usually '*' or an interface IP")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--topic", default="pose")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--replay_fps", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--source_joint_order", choices=["nmr", "isaaclab", "groot", "xml", "mujoco", "urdf"], default="nmr")
    parser.add_argument("--frame_index_source", choices=["sequential", "source"], default="sequential")
    parser.add_argument("--disable_catch_up", action="store_true")
    parser.add_argument("--window_size", type=int, default=50)
    parser.add_argument("--no_warmup_window", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with GrootZMQPublisher(
        bind_host=args.bind_host,
        port=args.port,
        topic=args.topic,
        fps=args.fps,
        source_joint_order=args.source_joint_order,
        frame_index_source=args.frame_index_source,
        catch_up=not bool(args.disable_catch_up),
        window_size=int(args.window_size),
        warmup_full_window=not bool(args.no_warmup_window),
    ) as publisher:
        print(f"[groot_zmq] publishing {args.npz} on {publisher.endpoint} topic={publisher.topic}", flush=True)
        replay_npz(
            npz_path=args.npz,
            publisher=publisher,
            batch_size=args.batch_size,
            replay_fps=args.replay_fps,
            start_index=args.start_index,
            max_frames=args.max_frames,
            loop=bool(args.loop),
        )
        print(json.dumps({"groot_zmq": publisher.meta()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
