#!/usr/bin/env python3
"""Run live camera or video-replay streaming inference with bounded queues."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class FramePacket:
    frame_id: int
    loop_idx: int
    capture_ts: float
    frame: np.ndarray


@dataclass
class BBoxPacket:
    frame_id: int
    loop_idx: int
    capture_ts: float
    frame: np.ndarray
    bbox_xyxy: np.ndarray
    score: float
    source: str


@dataclass
class FeaturePacket:
    frame_id: int
    loop_idx: int
    capture_ts: float
    feature: "torch.Tensor"


@dataclass
class ActionPacket:
    frame_id: int
    capture_ts: float
    commit_ts: float
    root_pos: np.ndarray
    root_rot_6d: np.ndarray
    dof: np.ndarray
    window_idx: int
    contact_prob: Optional[np.ndarray] = None
    contact_label: Optional[np.ndarray] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_source", choices=["video", "camera"], default="video")
    parser.add_argument("--video_path", default=None)
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "outputs/live_camera"))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--loops", type=int, default=10)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=None)
    parser.add_argument("--camera_index", type=int, default=0)
    parser.add_argument("--camera_device", default=None)
    parser.add_argument("--camera_frames", type=int, default=0, help="Maximum camera frames to process; <=0 runs until stopped.")
    parser.add_argument("--camera_width", type=int, default=1280)
    parser.add_argument("--camera_height", type=int, default=720)
    parser.add_argument("--camera_fourcc", default="MJPG")
    parser.add_argument("--camera_buffer_size", type=int, default=1)
    parser.add_argument("--enable_camera_preview", action="store_true")
    parser.add_argument("--preview_window_name", default="RGB2Robo camera")
    parser.add_argument("--preview_scale", type=float, default=1.0)
    parser.add_argument("--queue_size", type=int, default=320)
    parser.add_argument("--bbox_batch_size", type=int, default=2)
    parser.add_argument("--feature_batch_size", type=int, default=2)
    parser.add_argument("--flush_timeout", type=float, default=0.05)
    parser.add_argument(
        "--checkpoint",
        default=str(PROJECT_ROOT / "assets/checkpoints/rgb2robo_multirobot_clean.pth"),
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config/inference.yaml"),
    )
    parser.add_argument("--yolo_ckpt", default=str(PROJECT_ROOT / "stream_infer/assets/yolo11s.onnx"))
    parser.add_argument("--hmr2_ckpt", default=None, help="Defaults to assets/hmr2/epoch=10-step=25000.ckpt.")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--bbox_smooth_ema", type=float, default=0.7)
    parser.add_argument("--disable_bbox_smoothing", action="store_true")
    parser.add_argument("--disable_dense_timeline", action="store_true")
    parser.add_argument("--motion_stride", type=int, default=4)
    parser.add_argument("--action_queue_size", type=int, default=512)
    parser.add_argument("--action_target_buffer", type=int, default=40)
    parser.add_argument("--action_emit_mode", choices=["fifo", "lagged"], default="fifo")
    parser.add_argument("--action_emit_lag_sec", type=float, default=1.2)
    parser.add_argument("--drain_actions_fast", action="store_true")
    parser.add_argument("--action_emit_batch_size", type=int, default=20)
    parser.add_argument("--disable_action_realtime_skip", action="store_true")
    parser.add_argument("--disable_model_warmup", action="store_true")
    parser.add_argument("--warmup_batch_size", type=int, default=8)
    parser.add_argument("--disable_action_filter", action="store_true")
    parser.add_argument("--filter_root_alpha", type=float, default=0.7)
    parser.add_argument("--filter_rot_alpha", type=float, default=0.7)
    parser.add_argument("--filter_dof_alpha", type=float, default=0.45)
    parser.add_argument("--enable_contact_root_z", dest="enable_contact_root_z", action="store_true")
    parser.add_argument("--disable_contact_root_z", dest="enable_contact_root_z", action="store_false")
    parser.set_defaults(enable_contact_root_z=True)
    parser.add_argument("--contact_threshold", type=float, default=0.6)
    parser.add_argument("--contact_exit_threshold", type=float, default=0.35)
    parser.add_argument("--contact_enter_confirm_frames", type=int, default=2)
    parser.add_argument("--contact_release_hold_frames", type=int, default=2)
    parser.add_argument("--contact_root_z_alpha", type=float, default=0.5)
    parser.add_argument("--contact_root_z_max_offset", type=float, default=0.10)
    parser.add_argument("--contact_root_z_max_step", type=float, default=0.02)
    parser.add_argument("--disable_contact_root_xy", action="store_true")
    parser.add_argument("--contact_root_xy_speed_threshold_mps", type=float, default=1.0)
    parser.add_argument("--contact_root_xy_gain", type=float, default=0.5)
    parser.add_argument("--contact_root_xy_max_offset", type=float, default=0.04)
    parser.add_argument("--contact_root_xy_max_step", type=float, default=0.01)
    parser.add_argument("--xml_path", default=str(PROJECT_ROOT / "assets/robot/unitree_g1/g1_mocap_29dof.xml"))
    parser.add_argument("--enable_mujoco_viewer", action="store_true")
    parser.add_argument("--viewer_xml_path", default=None)
    parser.add_argument("--viewer_dof_order", choices=["mte_model", "xml"], default="mte_model")
    parser.add_argument("--viewer_start_timeout", type=float, default=5.0)
    parser.add_argument("--viewer_strict", action="store_true")
    parser.add_argument("--print_render_latency", action="store_true")
    parser.add_argument("--render_latency_print_every", type=int, default=30)
    parser.add_argument("--enable_groot_zmq", action="store_true")
    parser.add_argument("--groot_zmq_bind_host", default="*")
    parser.add_argument("--groot_zmq_port", type=int, default=5556)
    parser.add_argument("--groot_zmq_topic", default="pose")
    parser.add_argument("--groot_joint_order", choices=["nmr", "isaaclab", "groot", "xml", "mujoco", "urdf"], default="nmr")
    parser.add_argument("--groot_frame_index_source", choices=["sequential", "source"], default="sequential")
    parser.add_argument("--groot_zmq_send_hwm", type=int, default=10)
    parser.add_argument("--groot_zmq_window_size", type=int, default=50)
    parser.add_argument("--groot_zmq_publish_fps", type=float, default=50.0)
    parser.add_argument("--groot_zmq_no_warmup_window", action="store_true")
    parser.add_argument("--groot_zmq_start_delay_sec", type=float, default=0.2)
    parser.add_argument("--groot_disable_catch_up", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch

    from stream_infer.streaming_pipeline import (
        BBoxSmoother,
        HMR2FeatureStream,
        StreamMotionRunner,
        YoloBBoxStream,
        elapsed_since,
        now,
        read_video_info,
        save_stream_predictions,
        sync_cuda,
    )

    if args.input_source == "video":
        if not args.video_path:
            raise ValueError("--video_path is required when --input_source=video")
        video_path = Path(args.video_path).resolve()
    else:
        video_path = None
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.input_source == "video":
        info = read_video_info(video_path)
        video_frames = int(info["frames"])
        start_frame = max(0, int(args.start_frame))
        end_frame = video_frames if args.end_frame is None else min(video_frames, int(args.end_frame))
        if end_frame <= start_frame:
            raise ValueError(f"invalid frame range [{start_frame}, {end_frame}) for video with {video_frames} frames")
        segment_frames = end_frame - start_frame
        total_expected = int(segment_frames) * int(args.loops)
    else:
        start_frame = 0
        camera_frame_limit = int(args.camera_frames)
        if camera_frame_limit < 0:
            raise ValueError("--camera_frames must be >= 0 for camera input")
        segment_frames = camera_frame_limit
        end_frame = camera_frame_limit if camera_frame_limit > 0 else None
        total_expected = camera_frame_limit if camera_frame_limit > 0 else None
        info = {
            "path": str(args.camera_device if args.camera_device is not None else args.camera_index),
            "frames": None if total_expected is None else int(total_expected),
            "fps": float(args.fps),
            "width": int(args.camera_width),
            "height": int(args.camera_height),
            "input_source": "camera",
        }

    live_unbounded = args.input_source == "camera" and total_expected is None
    record_full_outputs = not live_unbounded

    groot_publisher = None
    if args.enable_groot_zmq:
        from stream_infer.groot_zmq_publisher import GrootZMQPublisher

        groot_publisher = GrootZMQPublisher(
            bind_host=str(args.groot_zmq_bind_host),
            port=int(args.groot_zmq_port),
            topic=str(args.groot_zmq_topic),
            fps=float(args.fps),
            source_joint_order=str(args.groot_joint_order),
            frame_index_source=str(args.groot_frame_index_source),
            catch_up=not bool(args.groot_disable_catch_up),
            window_size=int(args.groot_zmq_window_size),
            publish_fps=float(args.groot_zmq_publish_fps),
            warmup_full_window=not bool(args.groot_zmq_no_warmup_window),
            send_hwm=int(args.groot_zmq_send_hwm),
        )
        print(
            f"[groot_zmq] publishing GR00T protocol v1 on {groot_publisher.endpoint} "
            f"topic={groot_publisher.topic} window={groot_publisher.window_size} "
            f"alignment={groot_publisher.window_alignment} "
            f"source_fps={groot_publisher.fps:g} publish_fps={groot_publisher.publish_fps:g}",
            flush=True,
        )
        if float(args.groot_zmq_start_delay_sec) > 0.0:
            time.sleep(float(args.groot_zmq_start_delay_sec))

    yolo = YoloBBoxStream(args.yolo_ckpt, device="cuda", conf=args.conf)
    feature_stream = HMR2FeatureStream(
        args.hmr2_ckpt,
        device="cuda",
        batch_size=args.feature_batch_size,
    )
    motion_runner = StreamMotionRunner(args.config, args.checkpoint, device="cuda")

    def open_camera_capture() -> cv2.VideoCapture:
        camera_ref = args.camera_device if args.camera_device is not None else int(args.camera_index)
        if isinstance(camera_ref, int):
            cap = cv2.VideoCapture(camera_ref, cv2.CAP_V4L2)
        else:
            cap = cv2.VideoCapture(str(camera_ref), cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open camera: {camera_ref}")
        if args.camera_fourcc:
            fourcc = str(args.camera_fourcc)[:4]
            if len(fourcc) == 4:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(args.camera_width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(args.camera_height))
        cap.set(cv2.CAP_PROP_FPS, float(args.fps))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, int(args.camera_buffer_size))
        return cap

    warmup_meta: Dict[str, object] = {"enabled": not bool(args.disable_model_warmup)}
    if not args.disable_model_warmup:
        warmup_t0 = now()
        if args.input_source == "video":
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError(f"cannot open video for warmup: {video_path}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        else:
            cap = open_camera_capture()
        warmup_frames: List[np.ndarray] = []
        target_warmup = max(1, int(args.warmup_batch_size))
        while len(warmup_frames) < target_warmup:
            ok, frame = cap.read()
            if not ok:
                break
            warmup_frames.append(frame[:, :, ::-1])
        cap.release()
        if not warmup_frames:
            source_name = str(video_path) if args.input_source == "video" else str(args.camera_device or args.camera_index)
            raise RuntimeError(f"cannot read warmup frame from {args.input_source}: {source_name}")
        while len(warmup_frames) < target_warmup:
            warmup_frames.append(warmup_frames[-1].copy())

        # Force lazy modules and first CUDA kernels to initialize before the camera clock starts.
        feature_stream._ensure_model()
        warmup_ids = list(range(len(warmup_frames)))
        warmup_boxes, _ = yolo.detect_batch(warmup_frames, warmup_ids)
        warmup_xyxy = [np.asarray(rec.xyxy, dtype=np.float32) for rec in warmup_boxes]
        warmup_feats, _ = feature_stream.extract_batch(warmup_frames, warmup_xyxy)
        feature_dim = int(motion_runner.visual_cfg["input_dim"])
        dummy_window = torch.zeros(
            1,
            motion_runner.window_length,
            feature_dim,
            dtype=motion_runner.model_dtype,
            device=motion_runner.device,
        )
        dummy_valid = torch.ones(1, motion_runner.window_length, dtype=torch.bool, device=motion_runner.device)
        with torch.inference_mode():
            motion_runner.predict_window(dummy_window, valid_mask=dummy_valid)
        sync_cuda(torch)
        warmup_meta.update(
            {
                "sec": elapsed_since(warmup_t0, torch),
                "batch_size": int(len(warmup_frames)),
                "feature_shape": [int(x) for x in warmup_feats.shape],
            }
        )

    raw_queue: "Queue[Optional[FramePacket]]" = Queue(maxsize=args.queue_size)
    bbox_queue: "Queue[Optional[BBoxPacket]]" = Queue(maxsize=args.queue_size)
    feature_queue: "Queue[Optional[FeaturePacket]]" = Queue(maxsize=args.queue_size)
    action_queue: "Queue[Optional[ActionPacket]]" = Queue(maxsize=args.action_queue_size)
    visualization_fps = float(args.groot_zmq_publish_fps)
    mujoco_viewer = None
    if args.enable_mujoco_viewer:
        from stream_infer.live_mujoco_viewer import LiveMujocoViewer

        viewer_xml_path = str(args.viewer_xml_path or args.xml_path)
        mujoco_viewer = LiveMujocoViewer(
            viewer_xml_path,
            fps=visualization_fps,
            dof_order=str(args.viewer_dof_order),
        )
        if not mujoco_viewer.start(wait_sec=float(args.viewer_start_timeout), strict=bool(args.viewer_strict)):
            mujoco_viewer = None

    stage_log: List[Dict[str, object]] = []
    bbox_trace: List[Dict[str, object]] = []
    render_latency_values: List[float] = []
    live_recent_limit = 1000 if live_unbounded else 0
    recent_stats: Dict[str, int] = {
        "stage_log_total": 0,
        "bbox_trace_total": 0,
        "render_latency_total": 0,
        "emitted_action_total": 0,
    }

    def append_limited(store: List, item, counter_key: Optional[str] = None) -> None:
        if counter_key is not None:
            recent_stats[counter_key] = int(recent_stats.get(counter_key, 0)) + 1
        store.append(item)
        if live_recent_limit > 0 and len(store) > live_recent_limit:
            del store[: len(store) - live_recent_limit]

    def extend_limited(store: List, values) -> None:
        store.extend(values)
        if live_recent_limit > 0 and len(store) > live_recent_limit:
            del store[: len(store) - live_recent_limit]

    queue_stats = {
        "raw_enqueued": 0,
        "raw_dropped": 0,
        "bbox_enqueued": 0,
        "bbox_dropped": 0,
        "feature_enqueued": 0,
        "feature_dropped": 0,
        "action_enqueued": 0,
        "action_dropped": 0,
        "action_emitted": 0,
        "action_starved": 0,
        "action_realtime_skipped": 0,
        "max_raw_q": 0,
        "max_bbox_q": 0,
        "max_feature_q": 0,
        "max_action_q": 0,
    }
    error_holder: Dict[str, BaseException] = {}
    error_lock = threading.Lock()
    stop_event = threading.Event()
    bbox_smoother = BBoxSmoother(args.bbox_smooth_ema)
    smoother_loop_idx: Optional[int] = None

    def fail(exc: BaseException) -> None:
        with error_lock:
            if "exc" not in error_holder:
                error_holder["exc"] = exc
        stop_event.set()

    def put_drop_oldest(q: Queue, item, drop_key: str, max_key: str) -> None:
        while not stop_event.is_set():
            try:
                q.put_nowait(item)
                queue_stats[max_key] = max(queue_stats[max_key], q.qsize())
                return
            except Full:
                try:
                    q.get_nowait()
                    queue_stats[drop_key] += 1
                except Empty:
                    time.sleep(0.001)

    def camera_worker() -> None:
        frame_delay = 1.0 / max(float(args.fps), 1e-3)
        next_tick = time.perf_counter()
        preview_enabled = bool(args.enable_camera_preview)
        try:
            if preview_enabled:
                cv2.namedWindow(str(args.preview_window_name), cv2.WINDOW_NORMAL)
            if args.input_source == "video":
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    fail(RuntimeError(f"cannot open video: {video_path}"))
                    return
                loops = int(args.loops)
            else:
                cap = open_camera_capture()
                loops = 1
            try:
                for loop_idx in range(loops):
                    if args.input_source == "video":
                        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                    frame_idx = 0
                    while not stop_event.is_set():
                        if args.input_source == "video" and frame_idx >= segment_frames:
                            break
                        if args.input_source == "camera" and segment_frames > 0 and frame_idx >= segment_frames:
                            break
                        ok, frame = cap.read()
                        if not ok:
                            if args.input_source == "camera":
                                fail(RuntimeError("camera frame read failed"))
                            break
                        if preview_enabled:
                            preview_frame = frame
                            if abs(float(args.preview_scale) - 1.0) > 1e-3:
                                preview_frame = cv2.resize(
                                    frame,
                                    None,
                                    fx=float(args.preview_scale),
                                    fy=float(args.preview_scale),
                                    interpolation=cv2.INTER_AREA,
                                )
                            cv2.imshow(str(args.preview_window_name), preview_frame)
                            key = cv2.waitKey(1) & 0xFF
                            if key in (ord("q"), 27):
                                stop_event.set()
                                break
                        if args.input_source == "video":
                            frame_id = loop_idx * int(segment_frames) + frame_idx
                        else:
                            frame_id = frame_idx
                        packet = FramePacket(
                            frame_id=frame_id,
                            loop_idx=loop_idx,
                            capture_ts=time.perf_counter(),
                            frame=frame[:, :, ::-1],
                        )
                        put_drop_oldest(raw_queue, packet, "raw_dropped", "max_raw_q")
                        queue_stats["raw_enqueued"] += 1
                        frame_idx += 1
                        next_tick += frame_delay
                        sleep_for = next_tick - time.perf_counter()
                        if sleep_for > 0:
                            time.sleep(sleep_for)
            finally:
                cap.release()
        except BaseException as exc:  # noqa: BLE001
            fail(exc)
        finally:
            if preview_enabled:
                cv2.destroyWindow(str(args.preview_window_name))
            put_drop_oldest(raw_queue, None, "raw_dropped", "max_raw_q")

    def bbox_worker() -> None:
        buffer: List[FramePacket] = []
        try:
            while not stop_event.is_set():
                try:
                    item = raw_queue.get(timeout=args.flush_timeout)
                except Empty:
                    if buffer:
                        flush_bbox(buffer)
                    continue
                if item is None:
                    break
                buffer.append(item)
                if len(buffer) >= args.bbox_batch_size:
                    flush_bbox(buffer)
            if buffer:
                flush_bbox(buffer)
        except BaseException as exc:  # noqa: BLE001
            fail(exc)
        finally:
            put_drop_oldest(bbox_queue, None, "bbox_dropped", "max_bbox_q")

    def flush_bbox(buffer: List[FramePacket]) -> None:
        nonlocal smoother_loop_idx
        frame_ids = [pkt.frame_id for pkt in buffer]
        frames = [pkt.frame for pkt in buffer]
        records, timing = yolo.detect_batch(frames, frame_ids)
        append_limited(stage_log, {"stage": timing.stage, "sec": timing.sec, **timing.extra}, "stage_log_total")
        for pkt, rec in zip(buffer, records):
            raw_bbox = np.asarray(rec.xyxy, dtype=np.float32)
            if smoother_loop_idx is None or pkt.loop_idx != smoother_loop_idx:
                bbox_smoother.reset()
                smoother_loop_idx = pkt.loop_idx
            out_bbox = raw_bbox if args.disable_bbox_smoothing else bbox_smoother.update(raw_bbox)
            out_bbox = out_bbox.astype(np.float32)
            append_limited(
                bbox_trace,
                {
                    "frame_id": int(rec.frame_id),
                    "loop_idx": int(pkt.loop_idx),
                    "raw_xyxy": raw_bbox.tolist(),
                    "xyxy": out_bbox.tolist(),
                    "score": float(rec.score),
                    "source": str(rec.source),
                },
                "bbox_trace_total",
            )
            put_drop_oldest(
                bbox_queue,
                BBoxPacket(
                    frame_id=rec.frame_id,
                    loop_idx=pkt.loop_idx,
                    capture_ts=pkt.capture_ts,
                    frame=pkt.frame,
                    bbox_xyxy=out_bbox,
                    score=float(rec.score),
                    source=str(rec.source),
                ),
                "bbox_dropped",
                "max_bbox_q",
            )
            queue_stats["bbox_enqueued"] += 1
        buffer.clear()

    def feature_worker() -> None:
        buffer: List[BBoxPacket] = []
        try:
            while not stop_event.is_set():
                try:
                    item = bbox_queue.get(timeout=args.flush_timeout)
                except Empty:
                    if buffer:
                        flush_feature(buffer)
                    continue
                if item is None:
                    break
                buffer.append(item)
                if len(buffer) >= args.feature_batch_size:
                    flush_feature(buffer)
            if buffer:
                flush_feature(buffer)
        except BaseException as exc:  # noqa: BLE001
            fail(exc)
        finally:
            put_drop_oldest(feature_queue, None, "feature_dropped", "max_feature_q")

    def flush_feature(buffer: List[BBoxPacket]) -> None:
        frames = [pkt.frame for pkt in buffer]
        boxes = [pkt.bbox_xyxy for pkt in buffer]
        feats, timing = feature_stream.extract_batch(frames, boxes)
        append_limited(stage_log, {"stage": "hmr2_prep", "sec": timing["prep_sec"], "batch": len(buffer)}, "stage_log_total")
        append_limited(stage_log, {"stage": "hmr2_feature", "sec": timing["feature_sec"], "batch": len(buffer)}, "stage_log_total")
        for pkt, feat in zip(buffer, feats):
            put_drop_oldest(
                feature_queue,
                FeaturePacket(pkt.frame_id, pkt.loop_idx, pkt.capture_ts, feat.detach()),
                "feature_dropped",
                "max_feature_q",
            )
            queue_stats["feature_enqueued"] += 1
        buffer.clear()

    class StreamingMotionAccumulator:
        def __init__(self) -> None:
            self.model = motion_runner.model
            self.device = motion_runner.device
            self.model_dtype = motion_runner.model_dtype
            self.window_length = motion_runner.window_length
            self.step_size = max(1, int(args.motion_stride))
            if self.step_size > self.window_length:
                raise ValueError(f"motion_stride must be <= window_length ({self.window_length}), got {self.step_size}")
            self.action_tail = self.step_size
            self.context_span = motion_runner.context_span
            self.g1_dof = motion_runner.g1_dof
            self.feature_dim = int(motion_runner.visual_cfg["input_dim"])
            self.feature_store: List[torch.Tensor] = []
            self.capture_store: List[float] = []
            self.frame_id_store: List[int] = []
            self.window_means: List[torch.Tensor] = []
            self.next_start = 0
            self.next_action_index = max(0, self.window_length - self.action_tail)
            self.record_full_motion = bool(record_full_outputs)
            self.store_offset = 0
            self.total_features = 0
            self.window_count = 0
            self.log_limit = int(live_recent_limit)
            if self.record_full_motion:
                initial_capacity = int(total_expected) if total_expected is not None else max(1024, self.window_length * 4)
                self.accum = {
                    "root_pos": np.zeros((initial_capacity, 3), dtype=np.float32),
                    "root_rot_6d": np.zeros((initial_capacity, 6), dtype=np.float32),
                    "dof": np.zeros((initial_capacity, self.g1_dof), dtype=np.float32),
                }
                self.weight_sum = np.zeros((initial_capacity, 1), dtype=np.float32)
            else:
                self.accum = {}
                self.weight_sum = np.zeros((0, 1), dtype=np.float32)
            center = (self.window_length - 1) * 0.5
            dist = np.abs(np.arange(self.window_length, dtype=np.float32) - center)
            self.weights = np.clip(1.0 - dist / center, 1e-3, None) if center > 0 else np.ones(self.window_length, dtype=np.float32)
            self.motion_log: List[Dict[str, object]] = []
            self.motion_latency_log: List[Dict[str, object]] = []
            self.action_commit_log: List[Dict[str, object]] = []

        def _append_recent(self, store: List[Dict[str, object]], item: Dict[str, object]) -> None:
            store.append(item)
            if self.log_limit > 0 and len(store) > self.log_limit:
                del store[: len(store) - self.log_limit]

        def _ensure_motion_capacity(self, required_frames: int) -> None:
            if not self.record_full_motion:
                return
            current = int(self.weight_sum.shape[0])
            if required_frames <= current:
                return
            new_capacity = max(required_frames, current * 2 if current > 0 else self.window_length)
            for key, value in self.accum.items():
                pad_shape = (new_capacity - current, *value.shape[1:])
                self.accum[key] = np.concatenate([value, np.zeros(pad_shape, dtype=value.dtype)], axis=0)
            self.weight_sum = np.concatenate(
                [self.weight_sum, np.zeros((new_capacity - current, 1), dtype=self.weight_sum.dtype)],
                axis=0,
            )

        def add_feature(self, feature: torch.Tensor, frame_id: int, capture_ts: float) -> None:
            self.feature_store.append(feature.to(device=self.device, dtype=self.model_dtype, non_blocking=True))
            self.frame_id_store.append(int(frame_id))
            self.capture_store.append(float(capture_ts))
            self.total_features += 1
            self._consume_available(final=False)

        def finalize(self) -> None:
            self._consume_available(final=True)

        def _consume_available(self, final: bool) -> None:
            while self.next_start + self.window_length <= len(self.feature_store):
                self._run_window(self.next_start)
                self.next_start += self.step_size
            if final and len(self.feature_store) >= self.window_length:
                last_start = max(0, len(self.feature_store) - self.window_length)
                while self.next_start <= last_start:
                    self._run_window(self.next_start)
                    self.next_start += self.step_size
                if self.next_action_index < len(self.feature_store) and self.next_start < len(self.feature_store):
                    self._run_window(self.next_start)
                    self.next_start += self.step_size
            self._prune_live_history()

        def _prune_live_history(self) -> None:
            if self.record_full_motion:
                return
            remove_count = min(int(self.next_start), len(self.feature_store))
            if remove_count <= 0:
                return
            del self.feature_store[:remove_count]
            del self.capture_store[:remove_count]
            del self.frame_id_store[:remove_count]
            self.store_offset += remove_count
            self.next_start = max(0, self.next_start - remove_count)
            self.next_action_index = max(0, self.next_action_index - remove_count)

        def _build_context(self) -> Tuple[Optional[Dict[str, torch.Tensor]], Optional[torch.Tensor]]:
            if self.context_span <= 0 or not self.window_means:
                return None, None
            ctx = self.window_means[-self.context_span :]
            if not ctx:
                return None, None
            total_context = self.context_span
            kv = torch.zeros(total_context, self.feature_dim, dtype=self.model_dtype, device=self.device)
            mask = torch.zeros(total_context, dtype=torch.bool, device=self.device)
            count = min(len(ctx), total_context)
            kv[total_context - count :] = torch.stack(ctx[-count:], dim=0).to(device=self.device, dtype=self.model_dtype)
            mask[total_context - count :] = True
            return {"img_kv": kv.unsqueeze(0)}, mask.unsqueeze(0)

        def _run_window(self, start: int) -> None:
            end = min(start + self.window_length, len(self.feature_store))
            actual_length = end - start
            if actual_length <= 0:
                return
            window_idx = int(self.window_count)
            absolute_start = int(self.store_offset + start)
            absolute_end = int(self.store_offset + end)
            window_feat = torch.stack(self.feature_store[start:end], dim=0).to(device=self.device, dtype=self.model_dtype)
            if actual_length < self.window_length:
                pad = torch.zeros(self.window_length - actual_length, self.feature_dim, dtype=self.model_dtype, device=self.device)
                window_feat = torch.cat([window_feat, pad], dim=0)
            valid_mask = torch.zeros(1, self.window_length, dtype=torch.bool, device=self.device)
            valid_mask[:, :actual_length] = True
            context_kv, window_mask = self._build_context()
            if context_kv is not None:
                context_kv = {"img_kv": context_kv["img_kv"].to(device=self.device, non_blocking=True)}
                window_mask = window_mask.to(device=self.device, non_blocking=True)
            t0 = now()
            with torch.inference_mode():
                pred = motion_runner.predict_window(
                    window_feat.unsqueeze(0),
                    context_kv=context_kv,
                    window_mask=window_mask,
                    valid_mask=valid_mask,
                )
            out_ts = now()
            append_limited(
                stage_log,
                {
                    "stage": "motion_window",
                    "sec": elapsed_since(t0, torch),
                    "window_idx": window_idx,
                    "start": absolute_start,
                    "actual_length": actual_length,
                },
                "stage_log_total",
            )
            output = pred["motion"][0, :actual_length].detach().cpu().numpy().astype(np.float32)
            contact_prob = None
            contact_label = None
            if "contact_prob" in pred:
                contact_prob = pred["contact_prob"][0, :actual_length].detach().cpu().numpy().astype(np.float32)
            if "contact_label" in pred:
                contact_label = pred["contact_label"][0, :actual_length].detach().cpu().numpy().astype(np.int64)
            captures = self.capture_store[start:end]
            if captures:
                self._append_recent(
                    self.motion_latency_log,
                    {
                        "window_idx": window_idx,
                        "start": absolute_start,
                        "end": absolute_end,
                        "actual_length": actual_length,
                        "latency_first_frame_sec": float(out_ts - captures[0]),
                        "latency_last_frame_sec": float(out_ts - captures[-1]),
                        "latency_mean_frame_sec": float(out_ts - float(np.mean(captures))),
                    },
                )
            frame_weight = self.weights[:actual_length, None]
            if self.record_full_motion:
                self._ensure_motion_capacity(end)
                self.accum["root_pos"][start:end] += output[:, :3] * frame_weight
                self.accum["root_rot_6d"][start:end] += output[:, 3:9] * frame_weight
                self.accum["dof"][start:end] += output[:, 9 : 9 + self.g1_dof] * frame_weight
                self.weight_sum[start:end] += frame_weight
            if self.context_span > 0:
                self.window_means.append(window_feat[:actual_length].mean(dim=0).cpu())
                if len(self.window_means) > self.context_span:
                    del self.window_means[: len(self.window_means) - self.context_span]
            self._commit_tail_actions(start, end, output, contact_prob, contact_label, window_idx)
            self._append_recent(self.motion_log, {"start": absolute_start, "end": absolute_end})
            self.window_count += 1

        def _commit_tail_actions(
            self,
            start: int,
            end: int,
            output: np.ndarray,
            contact_prob: Optional[np.ndarray] = None,
            contact_label: Optional[np.ndarray] = None,
            window_idx: int = 0,
        ) -> None:
            tail_left = max(start, end - self.action_tail, self.next_action_index)
            if tail_left >= end:
                return
            commit_ts = time.perf_counter()
            for frame_index in range(tail_left, end):
                local_index = frame_index - start
                if local_index < 0 or local_index >= output.shape[0]:
                    continue
                packet = ActionPacket(
                    frame_id=int(self.frame_id_store[frame_index]),
                    capture_ts=float(self.capture_store[frame_index]),
                    commit_ts=float(commit_ts),
                    root_pos=output[local_index, :3].copy(),
                    root_rot_6d=output[local_index, 3:9].copy(),
                    dof=output[local_index, 9 : 9 + self.g1_dof].copy(),
                    window_idx=window_idx,
                    contact_prob=None if contact_prob is None else contact_prob[local_index].copy(),
                    contact_label=None if contact_label is None else contact_label[local_index].copy(),
                )
                put_drop_oldest(action_queue, packet, "action_dropped", "max_action_q")
                queue_stats["action_enqueued"] += 1
            self._append_recent(
                self.action_commit_log,
                {
                    "window_idx": window_idx,
                    "window_start": int(self.store_offset + start),
                    "window_end": int(self.store_offset + end),
                    "commit_start": int(self.store_offset + tail_left),
                    "commit_end": int(self.store_offset + end),
                    "count": int(end - tail_left),
                },
            )
            self.next_action_index = max(self.next_action_index, end)

        def final_motion(self) -> Dict[str, np.ndarray]:
            if not self.record_full_motion:
                return {
                    "root_pos": np.zeros((0, 3), dtype=np.float32),
                    "root_rot_6d": np.zeros((0, 6), dtype=np.float32),
                    "root_rot_quat": np.zeros((0, 4), dtype=np.float32),
                    "dof": np.zeros((0, self.g1_dof), dtype=np.float32),
                    "source_frame_ids": np.zeros((0,), dtype=np.int64),
                }
            valid = np.flatnonzero(self.weight_sum[:, 0] > 0)
            motion_len = int(valid[-1] + 1) if valid.size else 0
            denom = np.clip(self.weight_sum, 1e-6, None)
            root_pos = (self.accum["root_pos"] / denom)[:motion_len]
            root_rot_6d = (self.accum["root_rot_6d"] / denom)[:motion_len]
            dof = (self.accum["dof"] / denom)[:motion_len]
            from lib.util.robot_rotation import rot6d_to_quat_wxyz

            root_rot_quat = rot6d_to_quat_wxyz(torch.from_numpy(root_rot_6d)).cpu().numpy().astype(np.float32)
            return {
                "root_pos": root_pos,
                "root_rot_6d": root_rot_6d,
                "root_rot_quat": root_rot_quat,
                "dof": dof,
                "source_frame_ids": np.asarray(self.frame_id_store[:motion_len], dtype=np.int64),
            }
    motion_acc = StreamingMotionAccumulator()
    emitted_actions: List[Dict[str, object]] = []

    def motion_worker() -> None:
        try:
            while not stop_event.is_set():
                item = feature_queue.get()
                if item is None:
                    break
                motion_acc.add_feature(item.feature, item.frame_id, item.capture_ts)
            motion_acc.finalize()
            put_drop_oldest(action_queue, None, "action_dropped", "max_action_q")
        except BaseException as exc:  # noqa: BLE001
            fail(exc)

    class StreamingActionFilter:
        def __init__(self, root_alpha: float, rot_alpha: float, dof_alpha: float) -> None:
            self.root_alpha = float(np.clip(root_alpha, 0.0, 1.0))
            self.rot_alpha = float(np.clip(rot_alpha, 0.0, 1.0))
            self.dof_alpha = float(np.clip(dof_alpha, 0.0, 1.0))
            self.prev_root: Optional[np.ndarray] = None
            self.prev_rot6d: Optional[np.ndarray] = None
            self.prev_dof: Optional[np.ndarray] = None

        def update(self, item: ActionPacket) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            root = item.root_pos.astype(np.float32, copy=True)
            rot6d = item.root_rot_6d.astype(np.float32, copy=True)
            dof = item.dof.astype(np.float32, copy=True)
            if self.prev_root is None:
                self.prev_root = root
                self.prev_rot6d = rot6d
                self.prev_dof = dof
                return root.copy(), rot6d.copy(), dof.copy()

            root_out = self.root_alpha * root + (1.0 - self.root_alpha) * self.prev_root
            rot_out = self.rot_alpha * rot6d + (1.0 - self.rot_alpha) * self.prev_rot6d
            dof_out = self.dof_alpha * dof + (1.0 - self.dof_alpha) * self.prev_dof

            self.prev_root = root_out.astype(np.float32)
            self.prev_rot6d = rot_out.astype(np.float32)
            self.prev_dof = dof_out.astype(np.float32)
            return self.prev_root.copy(), self.prev_rot6d.copy(), self.prev_dof.copy()

    action_filter = None if args.disable_action_filter else StreamingActionFilter(
        args.filter_root_alpha,
        args.filter_rot_alpha,
        args.filter_dof_alpha,
    )
    contact_projector = None
    contact_project_stats: Dict[str, object] = {
        "enabled": False,
        "applied": 0,
        "delta_z": [],
    }
    if args.enable_contact_root_z:
        class StreamingRootZContactProjector:
            def __init__(self) -> None:
                from lib.robot.robot_kinematics import RobotKinematicsModel
                from lib.robot.robot_spec import load_robot_spec
                from lib.util.robot_rotation import rot6d_to_quat_wxyz
                from postprocess.robot_ik_postprocess import _foot_joint_specs, _marker_positions

                self.rot6d_to_quat_wxyz = rot6d_to_quat_wxyz
                self.marker_positions = _marker_positions
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                robot_spec = load_robot_spec(str(PROJECT_ROOT / "config" / "robot" / "g1.yaml"))
                self.kin = RobotKinematicsModel(
                    robot_spec.xml_path,
                    self.device,
                    model_to_xml_dof=lambda dof: __import__(
                        "lib.robot.robot_spec",
                        fromlist=["model_dof_to_xml_dof"],
                    ).model_dof_to_xml_dof(dof, robot_spec),
                    neutral_dof=robot_spec.neutral_dof,
                )
                foot_ids, foot_offsets = _foot_joint_specs(robot_spec, self.kin)
                self.foot_ids = foot_ids
                self.foot_offsets = torch.as_tensor(foot_offsets, dtype=torch.float32, device=self.device)
                self.active_contact = np.zeros((len(self.foot_ids),), dtype=bool)
                self.enter_count = np.zeros((len(self.foot_ids),), dtype=np.int32)
                self.release_count = np.zeros((len(self.foot_ids),), dtype=np.int32)
                self.previous_z_offset = 0.0
                self.previous_xy_offset = np.zeros((2,), dtype=np.float32)
                self.previous_foot_pos: Optional[np.ndarray] = None
                self.previous_support_mask = np.zeros((len(self.foot_ids),), dtype=bool)
                contact_project_stats.update(
                    {
                        "enabled": True,
                        "source": "model_contact_head",
                        "contact_enter_threshold": float(args.contact_threshold),
                        "contact_exit_threshold": float(args.contact_exit_threshold),
                        "contact_enter_confirm_frames": int(args.contact_enter_confirm_frames),
                        "contact_release_hold_frames": int(args.contact_release_hold_frames),
                        "root_z_alpha": float(args.contact_root_z_alpha),
                        "root_z_max_offset": float(args.contact_root_z_max_offset),
                        "root_z_max_step": float(args.contact_root_z_max_step),
                        "root_xy_enabled": not bool(args.disable_contact_root_xy),
                        "segment_frames": int(segment_frames),
                        "foot_body_ids": [int(x) for x in foot_ids],
                    }
                )

            def _model_contact_mask(self, contact_prob: Optional[np.ndarray]) -> np.ndarray:
                if contact_prob is None:
                    self.active_contact[:] = False
                    self.enter_count[:] = 0
                    self.release_count[:] = 0
                    return self.active_contact.copy()
                probabilities = np.asarray(contact_prob, dtype=np.float32).reshape(-1)
                for foot_idx in range(len(self.active_contact)):
                    value = float(probabilities[foot_idx]) if foot_idx < probabilities.size else 0.0
                    if self.active_contact[foot_idx]:
                        self.enter_count[foot_idx] = 0
                        if value >= float(args.contact_exit_threshold):
                            self.release_count[foot_idx] = 0
                        else:
                            self.release_count[foot_idx] += 1
                            if self.release_count[foot_idx] > int(args.contact_release_hold_frames):
                                self.active_contact[foot_idx] = False
                                self.release_count[foot_idx] = 0
                    elif value >= float(args.contact_threshold):
                        self.enter_count[foot_idx] += 1
                        if self.enter_count[foot_idx] >= max(1, int(args.contact_enter_confirm_frames)):
                            self.active_contact[foot_idx] = True
                            self.enter_count[foot_idx] = 0
                            self.release_count[foot_idx] = 0
                    else:
                        self.enter_count[foot_idx] = 0
                return self.active_contact.copy()

            def update(self, item: ActionPacket, root: np.ndarray, rot6d: np.ndarray, dof: np.ndarray) -> np.ndarray:
                return self.update_many([item], root.reshape(1, -1), rot6d.reshape(1, -1), dof.reshape(1, -1))[0]

            def update_many(
                self,
                items: List[ActionPacket],
                roots: np.ndarray,
                rot6ds: np.ndarray,
                dofs: np.ndarray,
            ) -> np.ndarray:
                roots_out = roots.astype(np.float32, copy=True)
                if len(items) == 0:
                    return roots_out
                foot_count = len(self.foot_ids)
                support_mask = np.zeros((len(items), foot_count), dtype=bool)
                active = np.zeros((len(items),), dtype=bool)
                for idx, item in enumerate(items):
                    support_mask[idx] = self._model_contact_mask(item.contact_prob)
                    active[idx] = bool(np.any(support_mask[idx]))
                if not bool(np.any(active)):
                    return roots_out
                active_idx = np.flatnonzero(active)
                root_t = torch.from_numpy(roots_out[active_idx]).float().to(self.device)
                rot6d_t = torch.from_numpy(rot6ds[active_idx].astype(np.float32, copy=False)).float().to(self.device)
                dof_t = torch.from_numpy(dofs[active_idx].astype(np.float32, copy=False)).float().to(self.device)
                quat_wxyz = self.rot6d_to_quat_wxyz(rot6d_t)
                quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
                with torch.inference_mode():
                    body_pos, body_rot = self.kin.forward_kinematics(root_t, quat_xyzw, dof_t)
                    foot_pos = self.marker_positions(body_pos, body_rot, self.foot_ids, self.foot_offsets)
                foot_positions = foot_pos.detach().float().cpu().numpy()
                foot_z = foot_positions[:, :, 2]
                target_offset = np.zeros((len(items),), dtype=np.float32)
                foot_by_item = {item_idx: foot_positions[local_idx] for local_idx, item_idx in enumerate(active_idx.tolist())}
                for local_idx, item_idx in enumerate(active_idx.tolist()):
                    target_offset[item_idx] = float(np.clip(-np.median(foot_z[local_idx, support_mask[item_idx]]), -float(args.contact_root_z_max_offset), float(args.contact_root_z_max_offset)))
                for idx in range(len(items)):
                    desired = float(target_offset[idx]) if active[idx] else 0.0
                    smoothed = float(args.contact_root_z_alpha) * desired + (1.0 - float(args.contact_root_z_alpha)) * self.previous_z_offset
                    step = float(np.clip(smoothed - self.previous_z_offset, -float(args.contact_root_z_max_step), float(args.contact_root_z_max_step)))
                    self.previous_z_offset += step
                    roots_out[idx, 2] += self.previous_z_offset
                    current = foot_by_item.get(idx)
                    if current is not None:
                        if not args.disable_contact_root_xy and self.previous_foot_pos is not None:
                            stable = support_mask[idx] & self.previous_support_mask
                            displacement = current[:, :2] - self.previous_foot_pos[:, :2]
                            stable &= np.linalg.norm(displacement, axis=-1) * float(args.fps) <= float(args.contact_root_xy_speed_threshold_mps)
                            if np.any(stable):
                                desired_xy = self.previous_xy_offset - float(args.contact_root_xy_gain) * np.median(displacement[stable], axis=0)
                                desired_xy *= min(1.0, float(args.contact_root_xy_max_offset) / max(float(np.linalg.norm(desired_xy)), 1e-8))
                                delta_xy = desired_xy - self.previous_xy_offset
                                delta_xy *= min(1.0, float(args.contact_root_xy_max_step) / max(float(np.linalg.norm(delta_xy)), 1e-8))
                                self.previous_xy_offset += delta_xy.astype(np.float32)
                        self.previous_foot_pos = current.copy()
                        self.previous_support_mask = support_mask[idx].copy()
                    else:
                        self.previous_foot_pos = None
                        self.previous_support_mask[:] = False
                    roots_out[idx, :2] += self.previous_xy_offset
                    if active[idx]:
                        contact_project_stats["applied"] = int(contact_project_stats["applied"]) + 1
                    append_limited(contact_project_stats["delta_z"], float(self.previous_z_offset))
                return roots_out

        contact_projector = StreamingRootZContactProjector()

    class ResampledMujocoVisualizer:
        def __init__(self, viewer, *, source_fps: float, output_fps: float) -> None:
            self.viewer = viewer
            self.source_fps = max(float(source_fps), 1e-6)
            self.output_fps = max(float(output_fps), 1e-6)
            self.resample_delay_sec = 1.0 / self.source_fps
            self.max_source_hold_sec = max(0.25, 4.0 / self.source_fps)
            self._samples: List[Tuple[float, np.ndarray, np.ndarray, np.ndarray, int, float]] = []
            self._sample_limit = 8
            self._lock = threading.Lock()
            self._stop_event = threading.Event()
            self._thread = threading.Thread(target=self._run, name="mujoco-visualizer-resampler", daemon=True)

        @staticmethod
        def _normalize_quat(quat: np.ndarray) -> np.ndarray:
            quat = np.asarray(quat, dtype=np.float32).reshape(4)
            return (quat / np.clip(np.linalg.norm(quat), 1e-6, None)).astype(np.float32)

        def start(self) -> None:
            self._thread.start()

        def close(self) -> None:
            self._stop_event.set()
            if self._thread.is_alive():
                self._thread.join(timeout=1.0)

        def submit_source_batch(
            self,
            *,
            roots: np.ndarray,
            quats: np.ndarray,
            dofs: np.ndarray,
            items: List[ActionPacket],
            source_ts: float,
        ) -> None:
            roots = np.asarray(roots, dtype=np.float32).reshape(len(items), 3)
            quats = np.asarray(quats, dtype=np.float32).reshape(len(items), 4)
            dofs = np.asarray(dofs, dtype=np.float32).reshape(len(items), -1)
            with self._lock:
                for idx, item in enumerate(items):
                    quat = self._normalize_quat(quats[idx])
                    if self._samples and float(np.dot(self._samples[-1][2], quat)) < 0.0:
                        quat = -quat
                    sample_time = float(source_ts) + float(idx) / self.source_fps
                    self._samples.append(
                        (
                            sample_time,
                            roots[idx].copy(),
                            quat.copy(),
                            dofs[idx].copy(),
                            int(item.frame_id),
                            float(item.capture_ts),
                        )
                    )
                    if len(self._samples) > self._sample_limit:
                        del self._samples[: len(self._samples) - self._sample_limit]

        def _run(self) -> None:
            period = 1.0 / self.output_fps
            next_tick = time.perf_counter()
            try:
                while not self._stop_event.is_set() and not stop_event.is_set():
                    next_tick += period
                    wait = next_tick - time.perf_counter()
                    if wait > 0.0:
                        self._stop_event.wait(wait)
                        if self._stop_event.is_set() or stop_event.is_set():
                            break
                    now_ts = time.perf_counter()
                    if next_tick < now_ts - period:
                        next_tick = now_ts
                    sample = self._interpolate(now_ts - self.resample_delay_sec, now_ts)
                    if sample is None:
                        continue
                    root, quat, dof, frame_id, capture_ts = sample
                    self._submit(root, quat, dof, frame_id, capture_ts)
            except BaseException as exc:  # noqa: BLE001
                fail(exc)

        def _interpolate(
            self,
            target_time: float,
            now_ts: float,
        ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, int, float]]:
            with self._lock:
                samples = list(self._samples)
            if not samples:
                return None
            latest_time, latest_root, latest_quat, latest_dof, latest_frame_id, latest_capture_ts = samples[-1]
            if now_ts - latest_time > self.max_source_hold_sec:
                return None
            if len(samples) == 1:
                return latest_root.copy(), latest_quat.copy(), latest_dof.copy(), latest_frame_id, latest_capture_ts

            prev_sample = samples[0]
            next_sample = samples[-1]
            for idx in range(1, len(samples)):
                if samples[idx][0] >= target_time:
                    prev_sample = samples[idx - 1]
                    next_sample = samples[idx]
                    break

            if target_time <= samples[0][0]:
                sample = samples[0]
                return sample[1].copy(), sample[2].copy(), sample[3].copy(), int(sample[4]), float(sample[5])
            if target_time >= samples[-1][0]:
                return latest_root.copy(), latest_quat.copy(), latest_dof.copy(), latest_frame_id, latest_capture_ts

            t0, root0, quat0, dof0, frame_id0, capture_ts0 = prev_sample
            t1, root1, quat1, dof1, frame_id1, capture_ts1 = next_sample
            alpha = float(np.clip((target_time - t0) / max(t1 - t0, 1e-9), 0.0, 1.0))
            root = ((1.0 - alpha) * root0 + alpha * root1).astype(np.float32)
            dof = ((1.0 - alpha) * dof0 + alpha * dof1).astype(np.float32)
            quat = self._normalize_quat(((1.0 - alpha) * quat0 + alpha * quat1).astype(np.float32))
            frame_id = int(round((1.0 - alpha) * float(frame_id0) + alpha * float(frame_id1)))
            capture_ts = float((1.0 - alpha) * float(capture_ts0) + alpha * float(capture_ts1))
            return root, quat, dof, frame_id, capture_ts

        def _submit(self, root: np.ndarray, quat: np.ndarray, dof: np.ndarray, frame_id: int, capture_ts: float) -> None:
            render_submit_ts = time.perf_counter()
            render_latency_sec = float(render_submit_ts - capture_ts)
            append_limited(render_latency_values, render_latency_sec, "render_latency_total")
            render_latency_count = int(recent_stats["render_latency_total"])
            if args.print_render_latency and (
                render_latency_count == 1
                or render_latency_count % max(1, int(args.render_latency_print_every)) == 0
            ):
                recent = np.asarray(render_latency_values[-max(1, int(args.render_latency_print_every)):], dtype=np.float32)
                print(
                    "[render_latency] "
                    f"frame={int(frame_id)} "
                    f"current={render_latency_sec * 1000.0:.1f}ms "
                    f"recent_mean={float(np.mean(recent)) * 1000.0:.1f}ms "
                    f"recent_p95={float(np.percentile(recent, 95)) * 1000.0:.1f}ms",
                    flush=True,
                )
            self.viewer.submit(root, quat, dof, frame_id=int(frame_id))

    mujoco_visualizer = None
    if mujoco_viewer is not None:
        mujoco_visualizer = ResampledMujocoVisualizer(
            mujoco_viewer,
            source_fps=float(args.fps),
            output_fps=visualization_fps,
        )
        mujoco_visualizer.start()
        print(
            f"[mujoco_viewer] visualization_fps={visualization_fps:g} source_fps={float(args.fps):g}",
            flush=True,
        )

    def append_emitted_actions(items: List[ActionPacket], emit_ts: float) -> None:
        if not items:
            return
        raw_roots = []
        raw_rot6ds = []
        raw_dofs = []
        filtered_roots = []
        filtered_rot6ds = []
        filtered_dofs = []
        for item in items:
            raw_root = item.root_pos.astype(np.float32)
            raw_rot6d = item.root_rot_6d.astype(np.float32)
            raw_dof = item.dof.astype(np.float32)
            if action_filter is None:
                filtered_root, filtered_rot6d, filtered_dof = raw_root.copy(), raw_rot6d.copy(), raw_dof.copy()
            else:
                filtered_root, filtered_rot6d, filtered_dof = action_filter.update(item)
            raw_roots.append(raw_root)
            raw_rot6ds.append(raw_rot6d)
            raw_dofs.append(raw_dof)
            filtered_roots.append(filtered_root)
            filtered_rot6ds.append(filtered_rot6d)
            filtered_dofs.append(filtered_dof)
        filtered_roots_arr = np.asarray(filtered_roots, dtype=np.float32)
        filtered_rot6ds_arr = np.asarray(filtered_rot6ds, dtype=np.float32)
        filtered_dofs_arr = np.asarray(filtered_dofs, dtype=np.float32)
        if contact_projector is not None:
            filtered_roots_arr = contact_projector.update_many(items, filtered_roots_arr, filtered_rot6ds_arr, filtered_dofs_arr)
        if groot_publisher is not None:
            try:
                groot_publisher.publish(
                    joint_pos=filtered_dofs_arr,
                    root_rot_6d=filtered_rot6ds_arr,
                    source_frame_ids=[int(item.frame_id) for item in items],
                )
            except BaseException as exc:  # noqa: BLE001
                fail(exc)
                return
        viewer_quats = None
        if mujoco_visualizer is not None:
            from lib.util.robot_rotation import rot6d_to_quat_wxyz

            viewer_quats = rot6d_to_quat_wxyz(torch.from_numpy(filtered_rot6ds_arr)).cpu().numpy().astype(np.float32)
        for idx, item in enumerate(items):
            append_limited(
                emitted_actions,
                {
                    "frame_id": int(item.frame_id),
                    "capture_ts": float(item.capture_ts),
                    "commit_ts": float(item.commit_ts),
                    "emit_ts": float(emit_ts),
                    "commit_latency_sec": float(item.commit_ts - item.capture_ts),
                    "latency_sec": float(emit_ts - item.capture_ts),
                    "root_pos_raw": raw_roots[idx],
                    "root_rot_6d_raw": raw_rot6ds[idx],
                    "dof_raw": raw_dofs[idx],
                    "root_pos": filtered_roots_arr[idx],
                    "root_rot_6d": filtered_rot6ds_arr[idx],
                    "dof": filtered_dofs_arr[idx],
                    "contact_prob": None if item.contact_prob is None else item.contact_prob.astype(np.float32),
                    "contact_label": None if item.contact_label is None else item.contact_label.astype(np.int64),
                    "window_idx": int(item.window_idx),
                },
                "emitted_action_total",
            )
        if mujoco_visualizer is not None and viewer_quats is not None:
            mujoco_visualizer.submit_source_batch(
                roots=filtered_roots_arr,
                quats=viewer_quats,
                dofs=filtered_dofs_arr,
                items=items,
                source_ts=emit_ts,
            )
        queue_stats["action_emitted"] += len(items)

    def append_emitted_action(item: ActionPacket, emit_ts: float) -> None:
        append_emitted_actions([item], emit_ts)

    def action_emitter_worker() -> None:
        frame_delay = 1.0 / max(float(args.fps), 1e-3)
        next_emit_tick: Optional[float] = None
        try:
            if args.action_emit_mode == "fifo":
                if args.drain_actions_fast:
                    batch: List[ActionPacket] = []

                    def flush_action_batch() -> None:
                        if batch:
                            append_emitted_actions(batch.copy(), time.perf_counter())
                            batch.clear()

                    while not stop_event.is_set():
                        try:
                            item = action_queue.get(timeout=frame_delay)
                        except Empty:
                            flush_action_batch()
                            continue
                        if item is None:
                            flush_action_batch()
                            break
                        batch.append(item)
                        if len(batch) >= max(1, int(args.action_emit_batch_size)):
                            flush_action_batch()
                    return
                while not stop_event.is_set():
                    if not args.disable_action_realtime_skip:
                        while action_queue.qsize() > int(args.action_target_buffer):
                            try:
                                dropped = action_queue.get_nowait()
                            except Empty:
                                break
                            if dropped is None:
                                put_drop_oldest(action_queue, None, "action_dropped", "max_action_q")
                                break
                            queue_stats["action_realtime_skipped"] += 1
                    try:
                        item = action_queue.get(timeout=frame_delay)
                    except Empty:
                        if next_emit_tick is not None:
                            queue_stats["action_starved"] += 1
                        continue
                    if item is None:
                        break
                    if next_emit_tick is None:
                        next_emit_tick = time.perf_counter()
                    sleep_for = next_emit_tick - time.perf_counter()
                    if sleep_for > 0:
                        time.sleep(sleep_for)
                    emit_ts = time.perf_counter()
                    append_emitted_action(item, emit_ts)
                    next_emit_tick += frame_delay
                return

            pending: Deque[ActionPacket] = deque()
            producer_done = False
            while not stop_event.is_set():
                while True:
                    try:
                        item = action_queue.get_nowait()
                    except Empty:
                        break
                    if item is None:
                        producer_done = True
                        continue
                    pending.append(item)

                now_ts = time.perf_counter()
                ready_cutoff = now_ts - float(args.action_emit_lag_sec)
                if pending and pending[0].capture_ts <= ready_cutoff:
                    item = pending.popleft()
                    if next_emit_tick is None:
                        next_emit_tick = now_ts
                    sleep_for = next_emit_tick - time.perf_counter()
                    if sleep_for > 0:
                        time.sleep(sleep_for)
                    emit_ts = time.perf_counter()
                    append_emitted_action(item, emit_ts)
                    next_emit_tick += frame_delay
                    continue

                if producer_done and not pending:
                    break
                try:
                    wait_for = frame_delay
                    if pending:
                        wait_for = max(0.0, min(frame_delay, pending[0].capture_ts + float(args.action_emit_lag_sec) - time.perf_counter()))
                    item = action_queue.get(timeout=wait_for if wait_for > 0 else frame_delay)
                except Empty:
                    if next_emit_tick is not None:
                        queue_stats["action_starved"] += 1
                    continue
                if item is None:
                    producer_done = True
                else:
                    pending.append(item)
        except BaseException as exc:  # noqa: BLE001
            fail(exc)

    t_total = now()
    decode_t = threading.Thread(target=camera_worker, daemon=True)
    bbox_t = threading.Thread(target=bbox_worker, daemon=True)
    feature_t = threading.Thread(target=feature_worker, daemon=True)
    motion_t = threading.Thread(target=motion_worker, daemon=True)
    action_t = threading.Thread(target=action_emitter_worker, daemon=True)
    decode_t.start()
    bbox_t.start()
    feature_t.start()
    motion_t.start()
    action_t.start()
    decode_t.join()
    bbox_t.join()
    feature_t.join()
    motion_t.join()
    action_t.join()
    sync_cuda(torch)
    total_sec = elapsed_since(t_total, torch)
    if mujoco_visualizer is not None:
        mujoco_visualizer.close()
    mujoco_viewer_meta = (
        {"enabled": False, "fps": None, "submitted": 0, "rendered": 0, "error": None}
        if mujoco_viewer is None
        else mujoco_viewer.stats()
    )
    if mujoco_viewer is not None:
        mujoco_viewer.close()
    groot_zmq_meta = {"enabled": False}
    if groot_publisher is not None:
        groot_publisher.close()
        groot_zmq_meta = groot_publisher.meta()

    if "exc" in error_holder:
        raise error_holder["exc"]

    motion = motion_acc.final_motion()
    processed_frames = int(motion_acc.total_features)
    motion_frames = int(motion["root_pos"].shape[0])

    def densify_motion_timeline(compact_motion: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
        source_ids = compact_motion.get("source_frame_ids")
        if args.disable_dense_timeline or source_ids is None or len(source_ids) == 0:
            return compact_motion, {"dense_timeline": False, "source_frame_gap_count": None}
        source_ids = np.asarray(source_ids, dtype=np.int64)
        gap_count = int(np.sum(np.diff(source_ids) > 1)) if len(source_ids) > 1 else 0
        dense_len = int(source_ids[-1] + 1)
        if dense_len == len(source_ids) and gap_count == 0:
            return compact_motion, {"dense_timeline": False, "source_frame_gap_count": 0}
        x_dense = np.arange(dense_len, dtype=np.float32)
        x_src = source_ids.astype(np.float32)
        dense: Dict[str, np.ndarray] = {}
        for key in ("root_pos", "root_rot_6d", "dof"):
            arr = compact_motion[key].astype(np.float32)
            flat = arr.reshape(arr.shape[0], -1)
            dense_flat = np.empty((dense_len, flat.shape[1]), dtype=np.float32)
            for col in range(flat.shape[1]):
                dense_flat[:, col] = np.interp(x_dense, x_src, flat[:, col]).astype(np.float32)
            dense[key] = dense_flat.reshape((dense_len, *arr.shape[1:]))
        from lib.util.robot_rotation import rot6d_to_quat_wxyz

        dense["root_rot_quat"] = rot6d_to_quat_wxyz(torch.from_numpy(dense["root_rot_6d"])).cpu().numpy().astype(np.float32)
        dense["source_frame_ids"] = np.arange(dense_len, dtype=np.int64)
        return dense, {
            "dense_timeline": True,
            "source_frame_gap_count": gap_count,
            "compact_motion_frames": int(len(source_ids)),
            "dense_motion_frames": int(dense_len),
        }

    save_motion, timeline_meta = densify_motion_timeline(motion)
    action_meta: Dict[str, object] = {
        "motion_stride": int(args.motion_stride),
        "action_queue_size": int(args.action_queue_size),
        "action_frames": int(recent_stats["emitted_action_total"]),
        "action_recent_frames": int(len(emitted_actions)),
        "action_record_full_outputs": bool(record_full_outputs),
    }
    if emitted_actions:
        action_frame_ids = np.asarray([x["frame_id"] for x in emitted_actions], dtype=np.int64)
        action_latency = np.asarray([x["latency_sec"] for x in emitted_actions], dtype=np.float32)
        action_root_pos = np.asarray([x["root_pos"] for x in emitted_actions], dtype=np.float32)
        action_root_rot_6d = np.asarray([x["root_rot_6d"] for x in emitted_actions], dtype=np.float32)
        action_dof = np.asarray([x["dof"] for x in emitted_actions], dtype=np.float32)
        action_root_pos_raw = np.asarray([x["root_pos_raw"] for x in emitted_actions], dtype=np.float32)
        action_root_rot_6d_raw = np.asarray([x["root_rot_6d_raw"] for x in emitted_actions], dtype=np.float32)
        action_dof_raw = np.asarray([x["dof_raw"] for x in emitted_actions], dtype=np.float32)
        action_emit_ts = np.asarray([x["emit_ts"] for x in emitted_actions], dtype=np.float64)
        action_commit_ts = np.asarray([x["commit_ts"] for x in emitted_actions], dtype=np.float64)
        action_capture_ts = np.asarray([x["capture_ts"] for x in emitted_actions], dtype=np.float64)
        action_commit_latency = np.asarray([x["commit_latency_sec"] for x in emitted_actions], dtype=np.float32)
        has_contact = all(x.get("contact_prob") is not None for x in emitted_actions)
        action_contact_prob = (
            np.asarray([x["contact_prob"] for x in emitted_actions], dtype=np.float32)
            if has_contact
            else np.zeros((len(emitted_actions), 0), dtype=np.float32)
        )
        action_contact_label = (
            np.asarray([x["contact_label"] for x in emitted_actions], dtype=np.int64)
            if has_contact and all(x.get("contact_label") is not None for x in emitted_actions)
            else (action_contact_prob > float(args.contact_threshold)).astype(np.int64)
        )
        from lib.util.robot_rotation import rot6d_to_quat_wxyz

        action_root_rot_quat = rot6d_to_quat_wxyz(torch.from_numpy(action_root_rot_6d)).cpu().numpy().astype(np.float32)
        action_root_rot_quat_raw = rot6d_to_quat_wxyz(torch.from_numpy(action_root_rot_6d_raw)).cpu().numpy().astype(np.float32)
        action_path = output_dir / "g1_action_stream.npz"
        action_raw_path = output_dir / "g1_action_stream_raw.npz"
        if record_full_outputs:
            np.savez_compressed(
                action_path,
                fps=np.array([float(args.fps)], dtype=np.float32),
                source_frame_ids=action_frame_ids,
                root_trans=action_root_pos,
                root_rot_6d=action_root_rot_6d,
                root_rot_quat=action_root_rot_quat,
                dof=action_dof,
                capture_ts=action_capture_ts,
                commit_ts=action_commit_ts,
                emit_ts=action_emit_ts,
                commit_latency_sec=action_commit_latency,
                latency_sec=action_latency,
                contact_prob=action_contact_prob,
                contact_label=action_contact_label,
            )
            np.savez_compressed(
                action_raw_path,
                fps=np.array([float(args.fps)], dtype=np.float32),
                source_frame_ids=action_frame_ids,
                root_trans=action_root_pos_raw,
                root_rot_6d=action_root_rot_6d_raw,
                root_rot_quat=action_root_rot_quat_raw,
                dof=action_dof_raw,
                capture_ts=action_capture_ts,
                commit_ts=action_commit_ts,
                emit_ts=action_emit_ts,
                commit_latency_sec=action_commit_latency,
                latency_sec=action_latency,
                contact_prob=action_contact_prob,
                contact_label=action_contact_label,
            )
        emit_duration = float(action_emit_ts[-1] - action_emit_ts[0]) if len(action_emit_ts) > 1 else 0.0

        def jump_stats(arr: np.ndarray) -> Dict[str, float]:
            if arr.shape[0] < 2:
                return {"mean": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
            jump = np.linalg.norm(np.diff(arr.reshape(arr.shape[0], -1), axis=0), axis=1)
            return {
                "mean": float(np.mean(jump)),
                "p95": float(np.percentile(jump, 95)),
                "p99": float(np.percentile(jump, 99)),
                "max": float(np.max(jump)),
            }

        filtered_jump = {
            "root_trans": jump_stats(action_root_pos),
            "root_rot_6d": jump_stats(action_root_rot_6d),
            "dof": jump_stats(action_dof),
        }
        raw_jump = {
            "root_trans": jump_stats(action_root_pos_raw),
            "root_rot_6d": jump_stats(action_root_rot_6d_raw),
            "dof": jump_stats(action_dof_raw),
        }
        action_meta.update(
            {
                "action_stream_path": str(action_path) if record_full_outputs else None,
                "action_stream_raw_path": str(action_raw_path) if record_full_outputs else None,
                "action_filter_enabled": not bool(args.disable_action_filter),
                "action_filter_root_alpha": None if args.disable_action_filter else float(args.filter_root_alpha),
                "action_filter_rot_alpha": None if args.disable_action_filter else float(args.filter_rot_alpha),
                "action_filter_dof_alpha": None if args.disable_action_filter else float(args.filter_dof_alpha),
                "action_has_contact": bool(has_contact),
                "action_first_frame_id": int(action_frame_ids[0]),
                "action_last_frame_id": int(action_frame_ids[-1]),
                "action_source_gap_count": int(np.sum(np.diff(action_frame_ids) > 1)) if len(action_frame_ids) > 1 else 0,
                "action_raw_jump": raw_jump,
                "action_filtered_jump": filtered_jump,
                "action_commit_latency_p50_sec": float(np.median(action_commit_latency)),
                "action_commit_latency_p95_sec": float(np.percentile(action_commit_latency, 95)),
                "action_commit_latency_mean_sec": float(np.mean(action_commit_latency)),
                "action_latency_p50_sec": float(np.median(action_latency)),
                "action_latency_p95_sec": float(np.percentile(action_latency, 95)),
                "action_latency_mean_sec": float(np.mean(action_latency)),
                "action_latency_min_sec": float(np.min(action_latency)),
                "action_latency_max_sec": float(np.max(action_latency)),
                "action_emit_duration_sec": emit_duration,
                "action_emit_fps": float((len(action_emit_ts) - 1) / emit_duration) if emit_duration > 0 else None,
            }
        )
    latency_meta: Dict[str, object] = {}
    if motion_acc.motion_latency_log:
        first_lat = [float(x["latency_first_frame_sec"]) for x in motion_acc.motion_latency_log]
        last_lat = [float(x["latency_last_frame_sec"]) for x in motion_acc.motion_latency_log]
        mean_lat = [float(x["latency_mean_frame_sec"]) for x in motion_acc.motion_latency_log]
        latency_meta = {
            "motion_latency_windows": motion_acc.motion_latency_log,
            "motion_latency_first_frame_p50_sec": float(np.median(first_lat)),
            "motion_latency_first_frame_p95_sec": float(np.percentile(first_lat, 95)),
            "motion_latency_last_frame_p50_sec": float(np.median(last_lat)),
            "motion_latency_last_frame_p95_sec": float(np.percentile(last_lat, 95)),
            "motion_latency_mean_frame_p50_sec": float(np.median(mean_lat)),
            "motion_latency_mean_frame_p95_sec": float(np.percentile(mean_lat, 95)),
        }
    contact_root_z_meta = dict(contact_project_stats)
    delta_z_values = np.asarray(contact_root_z_meta.pop("delta_z", []), dtype=np.float32)
    if delta_z_values.size:
        contact_root_z_meta.update(
            {
                "delta_z_mean": float(np.mean(delta_z_values)),
                "delta_z_p50": float(np.percentile(delta_z_values, 50)),
                "delta_z_p95": float(np.percentile(delta_z_values, 95)),
                "delta_z_min": float(np.min(delta_z_values)),
                "delta_z_max": float(np.max(delta_z_values)),
            }
        )
    render_latency_meta: Dict[str, object] = {
        "count": int(recent_stats["render_latency_total"]),
        "recent_count": int(len(render_latency_values)),
    }
    if render_latency_values:
        render_latency_arr = np.asarray(render_latency_values, dtype=np.float32)
        render_latency_meta.update(
            {
                "mean_sec": float(np.mean(render_latency_arr)),
                "p50_sec": float(np.percentile(render_latency_arr, 50)),
                "p95_sec": float(np.percentile(render_latency_arr, 95)),
                "min_sec": float(np.min(render_latency_arr)),
                "max_sec": float(np.max(render_latency_arr)),
            }
        )

    meta = {
        "input_source": str(args.input_source),
        "video_path": str(video_path),
        "camera_index": int(args.camera_index),
        "camera_device": None if args.camera_device is None else str(args.camera_device),
        "camera_frames": None if int(args.camera_frames) <= 0 else int(args.camera_frames),
        "camera_width": int(args.camera_width),
        "camera_height": int(args.camera_height),
        "camera_fourcc": str(args.camera_fourcc),
        "camera_buffer_size": int(args.camera_buffer_size),
        "loops": int(args.loops),
        "fps": float(args.fps),
        "start_frame": int(start_frame),
        "end_frame": None if end_frame is None else int(end_frame),
        "segment_frames": None if int(segment_frames) <= 0 else int(segment_frames),
        "queue_size": int(args.queue_size),
        "bbox_batch_size": int(args.bbox_batch_size),
        "feature_batch_size": int(args.feature_batch_size),
        "bbox_smooth_ema": None if args.disable_bbox_smoothing else float(args.bbox_smooth_ema),
        "motion_stride": int(args.motion_stride),
        "action_queue_size": int(args.action_queue_size),
        "action_target_buffer": int(args.action_target_buffer),
        "action_emit_mode": str(args.action_emit_mode),
        "action_emit_lag_sec": float(args.action_emit_lag_sec),
        "drain_actions_fast": bool(args.drain_actions_fast),
        "action_emit_batch_size": int(args.action_emit_batch_size),
        "action_realtime_skip": not bool(args.disable_action_realtime_skip),
        "action_filter_enabled": not bool(args.disable_action_filter),
        "action_filter_root_alpha": None if args.disable_action_filter else float(args.filter_root_alpha),
        "action_filter_rot_alpha": None if args.disable_action_filter else float(args.filter_rot_alpha),
        "action_filter_dof_alpha": None if args.disable_action_filter else float(args.filter_dof_alpha),
        "contact_root_z": contact_root_z_meta,
        "model_warmup": warmup_meta,
        "mujoco_viewer": mujoco_viewer_meta,
        "visualization_fps": float(visualization_fps),
        "groot_zmq": groot_zmq_meta,
        "render_latency_before_viewer": render_latency_meta,
        "live_unbounded": bool(live_unbounded),
        "record_full_outputs": bool(record_full_outputs),
        "recent_cache_limit": None if live_recent_limit <= 0 else int(live_recent_limit),
        "stage_log_total": int(recent_stats["stage_log_total"]),
        "stage_log_recent_count": int(len(stage_log)),
        "bbox_trace_total": int(recent_stats["bbox_trace_total"]),
        "bbox_trace_recent_count": int(len(bbox_trace)),
        "expected_frames": None if total_expected is None else int(total_expected),
        "processed_frames": int(processed_frames),
        "motion_frames": int(motion_frames),
        **action_meta,
        **timeline_meta,
        "raw_enqueued": int(queue_stats["raw_enqueued"]),
        "raw_dropped": int(queue_stats["raw_dropped"]),
        "bbox_enqueued": int(queue_stats["bbox_enqueued"]),
        "bbox_dropped": int(queue_stats["bbox_dropped"]),
        "feature_enqueued": int(queue_stats["feature_enqueued"]),
        "feature_dropped": int(queue_stats["feature_dropped"]),
        "action_enqueued": int(queue_stats["action_enqueued"]),
        "action_dropped": int(queue_stats["action_dropped"]),
        "action_emitted": int(queue_stats["action_emitted"]),
        "action_starved": int(queue_stats["action_starved"]),
        "action_realtime_skipped": int(queue_stats["action_realtime_skipped"]),
        "max_raw_q": int(queue_stats["max_raw_q"]),
        "max_bbox_q": int(queue_stats["max_bbox_q"]),
        "max_feature_q": int(queue_stats["max_feature_q"]),
        "max_action_q": int(queue_stats["max_action_q"]),
        "total_sec": total_sec,
        "effective_fps": processed_frames / total_sec if total_sec > 0 else None,
        "latency_first_motion_sec": float(motion_acc.motion_latency_log[0]["latency_first_frame_sec"]) if motion_acc.motion_latency_log else None,
        **latency_meta,
        "stages": stage_log,
    }
    if bbox_trace and record_full_outputs:
        np.savez_compressed(
            output_dir / "bbox_trace.npz",
            frame_id=np.asarray([r["frame_id"] for r in bbox_trace], dtype=np.int64),
            loop_idx=np.asarray([r["loop_idx"] for r in bbox_trace], dtype=np.int64),
            raw_xyxy=np.asarray([r["raw_xyxy"] for r in bbox_trace], dtype=np.float32),
            xyxy=np.asarray([r["xyxy"] for r in bbox_trace], dtype=np.float32),
            score=np.asarray([r["score"] for r in bbox_trace], dtype=np.float32),
            source=np.asarray([r["source"] for r in bbox_trace]),
        )
        meta["bbox_trace_path"] = str(output_dir / "bbox_trace.npz")
    if record_full_outputs and "source_frame_ids" in motion and len(motion["source_frame_ids"]) > 0:
        np.save(output_dir / "motion_source_frame_ids.npy", motion["source_frame_ids"])
        meta["motion_source_frame_ids_path"] = str(output_dir / "motion_source_frame_ids.npy")
    if record_full_outputs and "source_frame_ids" in save_motion and len(save_motion["source_frame_ids"]) > 0:
        np.save(output_dir / "saved_motion_source_frame_ids.npy", save_motion["source_frame_ids"])
        meta["saved_motion_source_frame_ids_path"] = str(output_dir / "saved_motion_source_frame_ids.npy")
    (output_dir / "live_camera_sim_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if record_full_outputs:
        save_stream_predictions(
            output_dir,
            fps=float(args.fps),
            motion=save_motion,
            extra_meta={
                "g1_xml_path": args.xml_path,
                "input_source": str(args.input_source),
                "video_path": str(video_path),
                "camera_index": int(args.camera_index),
                "camera_device": None if args.camera_device is None else str(args.camera_device),
                "start_frame": int(start_frame),
                "end_frame": None if end_frame is None else int(end_frame),
                "segment_frames": None if int(segment_frames) <= 0 else int(segment_frames),
                "config_path": args.config,
                "checkpoint_path": args.checkpoint,
                "lag_mode": "live_sim",
                "video_frames": int(save_motion["root_pos"].shape[0]),
                "video_fps": float(args.fps),
                "loops": int(args.loops),
                "queue_size": int(args.queue_size),
                "motion_stride": int(args.motion_stride),
                "action_queue_size": int(args.action_queue_size),
                "action_target_buffer": int(args.action_target_buffer),
                "action_emit_mode": str(args.action_emit_mode),
                "action_emit_lag_sec": float(args.action_emit_lag_sec),
                "drain_actions_fast": bool(args.drain_actions_fast),
                "action_emit_batch_size": int(args.action_emit_batch_size),
                "action_realtime_skip": not bool(args.disable_action_realtime_skip),
                "action_stream_path": action_meta.get("action_stream_path"),
                "action_stream_raw_path": action_meta.get("action_stream_raw_path"),
                "action_filter_enabled": not bool(args.disable_action_filter),
                "action_filter_root_alpha": None if args.disable_action_filter else float(args.filter_root_alpha),
                "action_filter_rot_alpha": None if args.disable_action_filter else float(args.filter_rot_alpha),
                "action_filter_dof_alpha": None if args.disable_action_filter else float(args.filter_dof_alpha),
                "groot_zmq": groot_zmq_meta,
                "visualization_fps": float(visualization_fps),
                "raw_dropped": int(queue_stats["raw_dropped"]),
                "bbox_smooth_ema": None if args.disable_bbox_smoothing else float(args.bbox_smooth_ema),
                **timeline_meta,
            },
        )
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
