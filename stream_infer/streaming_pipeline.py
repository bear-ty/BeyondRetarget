"""Streaming RGB video -> bbox -> HMR2 features -> RGB2Robo motion pipeline."""

from __future__ import annotations

import json
import os
import copy
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from lib.util.g1_export_utils import export_g1_predictions
from lib.util.robot_rotation import rot6d_to_quat_wxyz
from lib.util.visual_feature_config import resolve_visual_feature_cfg
from lib.util.window_inference import build_context_kv, compute_window_starts, to_plain_container


STREAM_INFER_ROOT = Path(__file__).resolve().parent
HMR2_SOURCE_ROOT = STREAM_INFER_ROOT.parent / "lib" / "vendor" / "hmr2"
DEFAULT_HMR2_CKPT = STREAM_INFER_ROOT.parent / "assets" / "hmr2" / "epoch=10-step=25000.ckpt"


def now() -> float:
    return time.perf_counter()


def sync_cuda(torch_module) -> None:
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def elapsed_since(start: float, torch_module=None) -> float:
    if torch_module is not None:
        sync_cuda(torch_module)
    return now() - start


def read_video_info(video_path: Path) -> Dict[str, object]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {
        "path": str(video_path),
        "frames": frames,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_sec": frames / fps if fps > 0 else None,
    }


def iter_video_frames(video_path: Path) -> Iterable[Tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame_id, frame[:, :, ::-1]
        frame_id += 1
    cap.release()


@dataclass
class BBoxRecord:
    frame_id: int
    xyxy: Tuple[float, float, float, float]
    score: float
    source: str


@dataclass
class TimingRecord:
    stage: str
    sec: float
    extra: Dict[str, object] = field(default_factory=dict)


class YoloBBoxStream:
    def __init__(self, yolo_ckpt: str, device: str = "cuda", conf: float = 0.35):
        from ultralytics import YOLO

        self.device = device
        self.use_half = str(device).startswith("cuda")
        self.backend = "ultralytics"
        self.conf = float(conf)
        self.yolo_ckpt = str(yolo_ckpt)
        self.model = None
        self._onnx_batch_size = None
        self.engine = None
        self._engine_trt = None
        self._engine_ctx = None
        self._engine_input_name = None
        self._engine_output_name = None
        self._engine_input = None
        self._engine_output = None
        self._engine_stream = None
        if self.yolo_ckpt.endswith(".engine"):
            if not Path(self.yolo_ckpt).is_file():
                raise FileNotFoundError(f"TensorRT engine not found: {self.yolo_ckpt}")
            self._load_engine(self.yolo_ckpt)
        else:
            self._select_onnx_device()
            self.model = YOLO(yolo_ckpt)
            try:
                self.model.to(self.device)
            except Exception:
                pass

    def _load_engine(self, engine_path: str) -> None:
        """Load a user-supplied TensorRT engine without bundling TensorRT."""
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise ImportError(
                "TensorRT is required only for --yolo_ckpt *.engine. "
                "Install a TensorRT version compatible with the engine or export an ONNX detector as described in stream_infer/README.md."
            ) from exc

        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as handle:
            runtime = trt.Runtime(logger)
            engine = runtime.deserialize_cuda_engine(handle.read())
        if engine is None:
            raise RuntimeError(f"failed to deserialize TensorRT engine: {engine_path}")
        self.engine = engine
        self._engine_trt = trt
        self._engine_ctx = engine.create_execution_context()
        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT detection requires CUDA")
        self._engine_stream = torch.cuda.Stream(device=self.device)
        io_names = [engine.get_tensor_name(index) for index in range(engine.num_io_tensors)]
        for name in io_names:
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._engine_input_name = name
            else:
                self._engine_output_name = name
        if self._engine_input_name is None or self._engine_output_name is None:
            raise RuntimeError(f"unexpected TensorRT engine I/O names: {io_names}")

    def _select_onnx_device(self) -> None:
        """Read the ONNX batch size and fall back to CPU when CUDA is unavailable."""
        if not self.yolo_ckpt.endswith(".onnx"):
            return
        wants_cuda = str(self.device).startswith("cuda")
        try:
            import onnxruntime as ort

            providers = ["CPUExecutionProvider"]
            if wants_cuda:
                providers.insert(0, "CUDAExecutionProvider")
            session = ort.InferenceSession(self.yolo_ckpt, providers=providers)
            batch_size = session.get_inputs()[0].shape[0]
            if isinstance(batch_size, int) and batch_size > 0:
                self._onnx_batch_size = batch_size
            available_providers = set(session.get_providers())
            del session
            if wants_cuda and "CUDAExecutionProvider" in available_providers:
                self.backend = "onnxruntime-cuda"
                return
        except Exception:
            pass
        self.device = "cpu"
        self.use_half = False
        self.backend = "onnxruntime-cpu-fallback" if wants_cuda else "onnxruntime-cpu"

    def detect_batch(self, frames: Sequence[np.ndarray], frame_ids: Sequence[int]) -> Tuple[List[BBoxRecord], TimingRecord]:
        t0 = now()
        if self.engine is not None:
            records = self._detect_batch_engine(frames, frame_ids)
            sync_cuda(torch)
            return records, TimingRecord("bbox_yolo_trt", elapsed_since(t0, torch), {"batch": len(frames), "backend": "tensorrt"})
        results = []
        batch_size = self._onnx_batch_size or max(1, len(frames))
        for start in range(0, len(frames), batch_size):
            batch = list(frames[start : start + batch_size])
            actual_count = len(batch)
            # Static ONNX inputs need a full batch, including during warmup and flushes.
            batch.extend([batch[-1]] * (batch_size - actual_count))
            predictions = self.model.predict(
                source=batch,
                device=self.device,
                conf=self.conf,
                classes=0,
                verbose=False,
                stream=False,
                half=self.use_half,
            )
            results.extend(predictions[:actual_count])
        sync_cuda(torch)
        records = self._records_from_ultralytics(results, frames, frame_ids)
        return records, TimingRecord("bbox_yolo", elapsed_since(t0, torch), {"batch": len(frames), "backend": self.backend})

    def _detect_batch_engine(self, frames: Sequence[np.ndarray], frame_ids: Sequence[int]) -> List[BBoxRecord]:
        if self.engine is None or self._engine_ctx is None:
            raise RuntimeError("TensorRT engine is not loaded")
        from ultralytics.data.augment import LetterBox
        from ultralytics.utils.ops import non_max_suppression, scale_boxes

        letterbox = LetterBox(new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, center=True)
        original_shapes = [frame.shape[:2] for frame in frames]
        images = [letterbox(image=frame) for frame in frames]
        images = np.stack(images, axis=0).astype(np.float32) / 255.0
        images = np.transpose(images, (0, 3, 1, 2))
        images = np.ascontiguousarray(images)
        input_tensor = torch.from_numpy(images).to(device=self.device, non_blocking=True)
        self._engine_ctx.set_input_shape(self._engine_input_name, tuple(input_tensor.shape))
        output_shape = tuple(self._engine_ctx.get_tensor_shape(self._engine_output_name))
        if self._engine_output is None or tuple(self._engine_output.shape) != output_shape:
            self._engine_output = torch.empty(output_shape, dtype=torch.float32, device=self.device)
        stream = self._engine_stream or torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            self._engine_input = input_tensor
            self._engine_ctx.set_tensor_address(self._engine_input_name, int(input_tensor.data_ptr()))
            self._engine_ctx.set_tensor_address(self._engine_output_name, int(self._engine_output.data_ptr()))
            self._engine_ctx.execute_async_v3(stream.cuda_stream)
        torch.cuda.current_stream().wait_stream(stream)
        detections = non_max_suppression(
            self._engine_output.detach().float(),
            conf_thres=self.conf,
            iou_thres=0.45,
            classes=[0],
            agnostic=False,
            max_det=1,
        )

        records: List[BBoxRecord] = []
        last_bbox: Optional[Tuple[float, float, float, float, float]] = None
        for frame_id, detection, frame, original_shape in zip(frame_ids, detections, frames, original_shapes):
            height, width = frame.shape[:2]
            if detection is None or detection.numel() == 0:
                bbox = last_bbox
            else:
                detection = detection[:1].clone()
                detection[:, :4] = scale_boxes((640, 640), detection[:, :4], original_shape)
                x1, y1, x2, y2, score, _ = detection[0].tolist()
                bbox = (float(x1), float(y1), float(x2), float(y2), float(score)) if x2 > x1 and y2 > y1 else last_bbox
            if bbox is None:
                margin_x, margin_y = width * 0.05, height * 0.05
                bbox = (margin_x, margin_y, width - margin_x, height - margin_y, 0.01)
                source = "full_fallback"
            else:
                source = "yolo_trt"
            last_bbox = bbox
            x1, y1, x2, y2, score = bbox
            records.append(BBoxRecord(int(frame_id), (float(x1), float(y1), float(x2), float(y2)), float(score), source))
        return records

    def _records_from_ultralytics(self, results, frames, frame_ids) -> List[BBoxRecord]:
        records: List[BBoxRecord] = []
        last_bbox: Optional[Tuple[float, float, float, float, float]] = None
        for frame_id, result, frame in zip(frame_ids, results, frames):
            h, w = frame.shape[:2]
            bbox = self._pick_person_bbox(result, last_bbox, w, h)
            if bbox is None:
                margin_x = w * 0.05
                margin_y = h * 0.05
                bbox = (margin_x, margin_y, w - margin_x, h - margin_y, 0.01)
                source = "full_fallback"
            else:
                source = "yolo"
            last_bbox = bbox
            x1, y1, x2, y2, score = bbox
            records.append(BBoxRecord(int(frame_id), (float(x1), float(y1), float(x2), float(y2)), float(score), source))
        return records

    @staticmethod
    def _pick_person_bbox(result, last_bbox, width: int, height: int) -> Optional[Tuple[float, float, float, float, float]]:
        try:
            pred = result.boxes
            if pred is None or len(pred) == 0:
                return last_bbox
            bboxes = pred.xyxy.detach().cpu().numpy()
            scores = pred.conf.detach().cpu().numpy()
            labels = pred.cls.detach().cpu().numpy().astype(np.int64)
            person_indices = np.where(labels == 0)[0]
            if person_indices.size > 0:
                best = person_indices[int(np.argmax(scores[person_indices]))]
            else:
                best = int(np.argmax(scores))
            x1, y1, x2, y2 = bboxes[best].tolist()
            if x2 <= x1 or y2 <= y1:
                return last_bbox
            return float(x1), float(y1), float(x2), float(y2), float(scores[best])
        except Exception:
            return last_bbox

class BBoxSmoother:
    def __init__(self, ema: float = 0.7):
        self.ema = float(ema)
        self.last_bbox: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.last_bbox = None

    def update(self, bbox_xyxy: np.ndarray) -> np.ndarray:
        bbox_xyxy = bbox_xyxy.astype(np.float32)
        if self.last_bbox is None:
            self.last_bbox = bbox_xyxy
        else:
            self.last_bbox = self.ema * self.last_bbox + (1.0 - self.ema) * bbox_xyxy
        return self.last_bbox.copy()


class HMR2FeatureStream:
    """HMR2 feature extractor backed by upstream ViT and Transformer modules."""

    def __init__(
        self,
        hmr2_ckpt: Optional[str] = None,
        device: str = "cuda",
        batch_size: int = 16,
    ):
        self.encoder_source_root = HMR2_SOURCE_ROOT
        self.hmr2_ckpt = str(Path(hmr2_ckpt).expanduser().resolve()) if hmr2_ckpt else str(DEFAULT_HMR2_CKPT)
        self.device = device
        self.batch_size = int(batch_size)
        if not Path(self.hmr2_ckpt).is_file():
            raise FileNotFoundError(f"HMR2 checkpoint not found: {self.hmr2_ckpt}")
        from lib.model.hmr2_encoder import load_hmr2

        self.load_hmr2 = load_hmr2
        self.model = None

    def _ensure_model(self):
        if self.model is None:
            self.model = self.load_hmr2(checkpoint_path=self.hmr2_ckpt).cuda().eval()
        return self.model

    def extract_batch(self, frames: Sequence[np.ndarray], bboxes_xyxy: Sequence[np.ndarray]) -> Tuple[torch.Tensor, Dict[str, float]]:
        if len(frames) != len(bboxes_xyxy):
            raise ValueError("frames and bboxes must have same length")
        model = self._ensure_model()
        from stream_infer.hmr2_preprocess import get_batch_from_frames, get_bbx_xys_from_xyxy

        t0 = now()
        bbx_xys = get_bbx_xys_from_xyxy(torch.from_numpy(np.stack(bboxes_xyxy, axis=0)).float(), base_enlarge=1.2).float()
        imgs = get_batch_from_frames(np.stack(frames, axis=0), bbx_xys)
        prep_sec = elapsed_since(t0, torch)

        t1 = now()
        feats: List[torch.Tensor] = []
        amp_enabled = torch.cuda.is_available()
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=amp_enabled, dtype=torch.float16):
            for j in range(0, imgs.shape[0], self.batch_size):
                batch = imgs[j : j + self.batch_size].cuda(non_blocking=True)
                feat = model({"img": batch})
                feats.append(feat.detach())
        feature_sec = elapsed_since(t1, torch)
        return torch.cat(feats, dim=0), {"prep_sec": prep_sec, "feature_sec": feature_sec}


class StreamMotionRunner:
    def __init__(self, config_path: str, checkpoint_path: str, device: str = "cuda"):
        from omegaconf import OmegaConf

        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.cfg = OmegaConf.load(config_path)
        self.task = copy.deepcopy(self.cfg.task if "task" in self.cfg else self.cfg)
        from postprocess.contact_infer import load_contact_head
        from scripts.run_multirobot_pipeline import checkpoint_robot_heads, load_unified_model

        robot_heads = checkpoint_robot_heads(checkpoint_path, requested=["g1"])
        self.g1_dof = int(robot_heads["g1"]["dof"])
        self.task["g1_dof"] = self.g1_dof
        self.model = load_unified_model(config_path, checkpoint_path, self.device, robot_heads)
        self.contact_model = load_contact_head(checkpoint_path, self.device)
        self.load_info = {"unified_robot_heads": robot_heads}
        self.model_dtype = next(self.model.parameters()).dtype
        self.visual_cfg = resolve_visual_feature_cfg(to_plain_container(self.task.get("visual_feature", {})))
        self.window_length = int(self.task.get("window_length", 50))
        self.overlap_frames = int(self.task.get("cross_window_config", {}).get("overlap_frames", 10))
        self.context_span = int(self.task.get("cross_window_config", {}).get("context_span", 8))
        self.bidirectional = bool(self.task.get("cross_window_config", {}).get("attention", {}).get("bidirectional", True))
        self.step_size = max(1, self.window_length - self.overlap_frames)

    def predict_window(
        self,
        window_feat: torch.Tensor,
        context_kv: Optional[Dict[str, torch.Tensor]] = None,
        window_mask: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if self.contact_model is not None and hasattr(self.model, "extract_motion_feature"):
            from postprocess.contact_infer import predict_contact_prob

            motion_feature = self.model.extract_motion_feature(
                window_feat,
                context_kv=context_kv,
                window_mask=window_mask,
                valid_mask=valid_mask,
            )
            pred = self.model.predict_from_feature(
                motion_feature,
                int(window_feat.shape[0]),
                int(window_feat.shape[1]),
                "g1",
                valid_mask=valid_mask,
            )
            contact_out = predict_contact_prob(
                self.contact_model,
                motion_feature.view(int(window_feat.shape[0]), int(window_feat.shape[1]), -1),
                window_feat,
                valid_mask=valid_mask,
            )
            pred["contact_logits"] = contact_out["logits"]
            pred["contact_prob"] = contact_out["prob"]
            pred["contact_label"] = contact_out["label"]
            return pred
        return self.model(
            window_feat,
            context_kv=context_kv,
            window_mask=window_mask,
            valid_mask=valid_mask,
        )

    def run_windows(
        self,
        feature_tensor: torch.Tensor,
        fps: float,
        lag_mode: str = "causal",
    ) -> Tuple[Dict[str, np.ndarray], List[TimingRecord]]:
        seq_len = int(feature_tensor.shape[0])
        feature_dim = int(feature_tensor.shape[-1])
        starts = compute_window_starts(seq_len, self.window_length, self.step_size)
        weights = np.ones(self.window_length, dtype=np.float32)
        if self.window_length > 1:
            center = (self.window_length - 1) * 0.5
            dist = np.abs(np.arange(self.window_length, dtype=np.float32) - center)
            weights = np.clip(1.0 - dist / center, 1e-3, None)

        accum = {
            "root_pos": np.zeros((seq_len, 3), dtype=np.float32),
            "root_rot_6d": np.zeros((seq_len, 6), dtype=np.float32),
            "dof": np.zeros((seq_len, self.g1_dof), dtype=np.float32),
        }
        weight_sum = np.zeros((seq_len, 1), dtype=np.float32)
        timings: List[TimingRecord] = []

        with torch.inference_mode():
            for win_idx, left in enumerate(starts):
                right = min(seq_len, left + self.window_length)
                actual_length = right - left
                if actual_length <= 0:
                    continue
                feature = torch.zeros(1, self.window_length, feature_dim, dtype=feature_tensor.dtype)
                feature[0, :actual_length] = feature_tensor[left:right]
                valid_mask = torch.zeros(1, self.window_length, dtype=torch.bool, device=self.device)
                valid_mask[:, :actual_length] = True
                context_kv, window_mask = self._build_stream_context(feature_tensor, starts, win_idx, lag_mode)
                if context_kv is not None:
                    context_kv = {"img_kv": context_kv["img_kv"].to(device=self.device, non_blocking=True)}
                    window_mask = window_mask.to(device=self.device, non_blocking=True)
                t0 = now()
                pred = self.predict_window(
                    feature.to(device=self.device, non_blocking=True),
                    context_kv=context_kv,
                    window_mask=window_mask,
                    valid_mask=valid_mask,
                )
                timings.append(TimingRecord("motion_window", elapsed_since(t0, torch), {"window_idx": win_idx, "start": left, "actual_length": actual_length}))
                output = pred["motion"][0, :actual_length].detach().cpu().numpy().astype(np.float32)
                frame_weight = weights[:actual_length, None]
                accum["root_pos"][left:right] += output[:, :3] * frame_weight
                accum["root_rot_6d"][left:right] += output[:, 3:9] * frame_weight
                accum["dof"][left:right] += output[:, 9 : 9 + self.g1_dof] * frame_weight
                weight_sum[left:right] += frame_weight

        weight_sum = np.clip(weight_sum, 1e-6, None)
        root_pos = accum["root_pos"] / weight_sum
        root_rot_6d = accum["root_rot_6d"] / weight_sum
        dof = accum["dof"] / weight_sum
        root_rot_quat = rot6d_to_quat_wxyz(torch.from_numpy(root_rot_6d)).cpu().numpy().astype(np.float32)
        return (
            {
                "root_pos": root_pos,
                "root_rot_6d": root_rot_6d,
                "root_rot_quat": root_rot_quat,
                "dof": dof,
            },
            timings,
        )

    def _build_stream_context(
        self,
        feature_tensor: torch.Tensor,
        starts: List[int],
        current_idx: int,
        lag_mode: str,
    ) -> Tuple[Optional[Dict[str, torch.Tensor]], Optional[torch.Tensor]]:
        if lag_mode == "bidirectional":
            return build_context_kv(feature_tensor, starts, current_idx, self.context_span, self.bidirectional, self.window_length)

        seq_len = int(feature_tensor.shape[0])
        ctx_feats: List[torch.Tensor] = []
        start_prev = max(0, current_idx - self.context_span)
        for win_idx in range(start_prev, current_idx):
            left = starts[win_idx]
            right = min(seq_len, left + self.window_length)
            ctx_feats.append(feature_tensor[left:right].mean(dim=0))
        total_context = self.context_span if not self.bidirectional else self.context_span * 2
        if not ctx_feats:
            return None, None
        kv = feature_tensor.new_zeros(total_context, feature_tensor.shape[-1])
        mask = torch.zeros(total_context, dtype=torch.bool, device=feature_tensor.device)
        count = len(ctx_feats)
        # Keep past context next to the current window; future slots remain masked.
        kv[self.context_span - count : self.context_span] = torch.stack(ctx_feats, dim=0)
        mask[self.context_span - count : self.context_span] = True
        return {"img_kv": kv.unsqueeze(0)}, mask.unsqueeze(0)


def save_stream_predictions(
    output_dir: Path,
    fps: float,
    motion: Dict[str, np.ndarray],
    extra_meta: Dict[str, object],
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pred = {
        "fps": np.array([float(fps)], dtype=np.float32),
        "root_trans": motion["root_pos"].astype(np.float32),
        "root_rot_quat": motion["root_rot_quat"].astype(np.float32),
        "root_rot_6d": motion["root_rot_6d"].astype(np.float32),
        "dof": motion["dof"].astype(np.float32),
    }
    export_g1_predictions(output_dir, pred, str(extra_meta["g1_xml_path"]))
    meta_path = output_dir / "stream_meta.json"
    meta_path.write_text(json.dumps(extra_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"meta_path": str(meta_path)}
