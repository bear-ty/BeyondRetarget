#!/usr/bin/env python3
"""Live MuJoCo viewer sink for streamed G1 robot actions."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


G1_MTE_TO_XML_JOINT_MAPPING = np.asarray(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
    dtype=np.int64,
)


def map_dof_to_xml_order(dof: np.ndarray, dof_order: str = "mte_model") -> np.ndarray:
    dof = np.asarray(dof, dtype=np.float64)
    if dof_order == "xml":
        return dof
    if dof_order == "mte_model":
        mapped = np.zeros_like(dof)
        mapped[..., G1_MTE_TO_XML_JOINT_MAPPING] = dof
        return mapped
    raise ValueError(f"Unsupported dof_order: {dof_order}")


class LiveMujocoViewer:
    """Passive MuJoCo viewer sink for streamed actions."""

    def __init__(
        self,
        xml_path: str,
        fps: float = 30.0,
        dof_order: str = "mte_model",
        title: str = "RGB2Robo live G1",
        camera_distance: float = 2.4,
        camera_azimuth: float = 160.0,
        camera_elevation: float = -12.0,
    ) -> None:
        self.xml_path = str(xml_path)
        self.fps = float(fps)
        self.dof_order = str(dof_order)
        self.title = str(title)
        self.camera_distance = float(camera_distance)
        self.camera_azimuth = float(camera_azimuth)
        self.camera_elevation = float(camera_elevation)
        self._error: Optional[BaseException] = None
        self._mj = None
        self._model = None
        self._data = None
        self._viewer = None
        self._last_sync = 0.0
        self.submitted = 0
        self.rendered = 0

    def start(self, wait_sec: float = 5.0, strict: bool = False) -> bool:
        del wait_sec
        if self._viewer is not None:
            return self._error is None
        try:
            import mujoco as mj
            import mujoco.viewer

            self._mj = mj
            self._model = mj.MjModel.from_xml_path(self.xml_path)
            self._data = mj.MjData(self._model)
            self._viewer = mujoco.viewer.launch_passive(
                self._model,
                self._data,
                show_left_ui=False,
                show_right_ui=False,
            )
            self._viewer.cam.type = mj.mjtCamera.mjCAMERA_FREE
            self._viewer.cam.distance = self.camera_distance
            self._viewer.cam.azimuth = self.camera_azimuth
            self._viewer.cam.elevation = self.camera_elevation
            self._viewer.sync()
            self._last_sync = time.perf_counter()
            return True
        except BaseException as exc:
            self._error = exc
            if strict:
                raise RuntimeError("MuJoCo viewer failed to start") from self._error
            print(f"[mujoco_viewer] disabled: {self._error}", file=sys.stderr)
            return False

    def submit(
        self,
        root_pos: np.ndarray,
        root_rot_wxyz: np.ndarray,
        dof: np.ndarray,
        frame_id: int = -1,
    ) -> None:
        root_pos = np.asarray(root_pos, dtype=np.float64).reshape(3).copy()
        root_rot_wxyz = np.asarray(root_rot_wxyz, dtype=np.float64).reshape(4).copy()
        quat_norm = float(np.linalg.norm(root_rot_wxyz))
        if quat_norm > 1e-8:
            root_rot_wxyz /= quat_norm
        dof_xml = map_dof_to_xml_order(np.asarray(dof).reshape(-1), self.dof_order).copy()
        self.submitted += 1
        self._apply(root_pos, root_rot_wxyz, dof_xml, int(frame_id))

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def stats(self) -> dict:
        return {
            "enabled": self._viewer is not None and self._error is None,
            "fps": float(self.fps),
            "submitted": int(self.submitted),
            "rendered": int(self.rendered),
            "error": None if self._error is None else repr(self._error),
        }

    def _apply(self, root_pos: np.ndarray, root_rot_wxyz: np.ndarray, dof_xml: np.ndarray, frame_id: int) -> None:
        del frame_id
        if self._viewer is None or self._model is None or self._data is None or self._mj is None:
            return
        if not self._viewer.is_running():
            self.close()
            return
        mj = self._mj
        model = self._model
        data = self._data
        if model.nq >= 7 and model.jnt_type[0] == mj.mjtJoint.mjJNT_FREE:
            data.qpos[:3] = root_pos
            data.qpos[3:7] = root_rot_wxyz
            data.qpos[7 : 7 + min(dof_xml.shape[0], model.nq - 7)] = dof_xml[: model.nq - 7]
        else:
            data.qpos[: min(dof_xml.shape[0], model.nq)] = dof_xml[: model.nq]
        mj.mj_forward(model, data)
        self.rendered += 1
        if model.nbody > 1:
            self._viewer.cam.lookat[:] = data.xpos[1]
        now_ts = time.perf_counter()
        min_delay = 1.0 / max(self.fps, 1e-3)
        if now_ts - self._last_sync >= min_delay * 0.5:
            self._viewer.sync()
            self._last_sync = now_ts


def _load_npz_motion(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, str]:
    data = np.load(path, allow_pickle=True)
    required = {"root_trans", "root_rot_quat", "dof"}
    missing = required.difference(data.files)
    if missing:
        raise RuntimeError(f"{path} missing keys: {sorted(missing)}")
    fps = float(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data.files else 30.0
    dof_order = str(np.asarray(data["dof_order"]).reshape(-1)[0]) if "dof_order" in data.files else "mte_model"
    return (
        np.asarray(data["root_trans"], dtype=np.float32),
        np.asarray(data["root_rot_quat"], dtype=np.float32),
        np.asarray(data["dof"], dtype=np.float32),
        fps,
        dof_order,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a saved RGB2Robo G1 action stream in a live MuJoCo window.")
    parser.add_argument("action_stream", help="Path to g1_action_stream.npz")
    parser.add_argument("--xml_path", default=str(PROJECT_ROOT / "assets" / "robot" / "unitree_g1" / "g1_mocap_29dof.xml"))
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--dof_order", default=None, choices=["mte_model", "xml"])
    parser.add_argument("--loop", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots, quats, dofs, file_fps, file_dof_order = _load_npz_motion(Path(args.action_stream))
    fps = float(args.fps if args.fps is not None else file_fps)
    dof_order = str(args.dof_order if args.dof_order is not None else file_dof_order)
    viewer = LiveMujocoViewer(args.xml_path, fps=fps, dof_order=dof_order)
    viewer.start(strict=True)
    try:
        frame_delay = 1.0 / max(fps, 1e-3)
        while True:
            for frame_idx in range(roots.shape[0]):
                viewer.submit(roots[frame_idx], quats[frame_idx], dofs[frame_idx], frame_idx)
                time.sleep(frame_delay)
            if not args.loop:
                break
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
