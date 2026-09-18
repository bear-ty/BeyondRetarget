#!/usr/bin/env python3
"""Check whether an RGB2Robo installation is ready for inference."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import os
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.download_rgb2robo_assets import ASSETS, validate_asset_file  # noqa: E402


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


EXPECTED_VERSIONS = {
    "torch": "2.3.0",
    "torchvision": "0.18.0",
    "ultralytics": "8.3.0",
}
IMPORTS = {
    "hydra-core": "hydra",
    "loguru": "loguru",
    "numpy": "numpy",
    "onnxruntime": "onnxruntime",
    "omegaconf": "omegaconf",
    "opencv-python": "cv2",
    "pytorch-lightning": "pytorch_lightning",
    "scipy": "scipy",
    "tensorboard": "tensorboard",
    "timm": "timm",
    "tqdm": "tqdm",
    "trimesh": "trimesh",
    "ultralytics": "ultralytics",
    "yacs": "yacs",
    "einops": "einops",
    "imageio": "imageio",
    "av": "av",
    "ffmpeg-python": "ffmpeg",
    "mujoco": "mujoco",
    "pytorch3d": "pytorch3d",
    "pyzmq": "zmq",
    "smplx": "smplx",
    "lapx": "lap",
}


class CheckReport:
    def __init__(self) -> None:
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def ok(self, message: str) -> None:
        print(f"[ok] {message}")

    def error(self, message: str) -> None:
        self.errors.append(message)
        print(f"[error] {message}")

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        print(f"[warning] {message}")


def installed_version(distribution: str) -> str:
    return importlib.metadata.version(distribution)


def check_environment(report: CheckReport, allow_no_cuda: bool) -> None:
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "AMD64"}:
        report.error(f"unsupported platform: {platform.system()} {platform.machine()}; expected Linux x86_64")
    else:
        report.ok("platform is Linux x86_64")

    if sys.version_info[:2] != (3, 10):
        report.error(f"Python {sys.version.split()[0]} is active; expected Python 3.10")
    else:
        report.ok(f"Python {sys.version.split()[0]}")

    for distribution, module_name in IMPORTS.items():
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            report.error(f"cannot import {module_name} ({distribution}): {exc}")
        else:
            report.ok(f"import {module_name}")

    for distribution, expected in EXPECTED_VERSIONS.items():
        try:
            actual = installed_version(distribution)
        except importlib.metadata.PackageNotFoundError:
            report.error(f"distribution {distribution} is not installed")
            continue
        if actual != expected and not actual.startswith(expected + "+"):
            report.error(f"{distribution}=={actual}; expected {expected}")
        else:
            report.ok(f"{distribution}=={actual}")

    try:
        from ultralytics.nn.modules.block import C3k2  # noqa: F401
    except Exception as exc:
        report.error(f"Ultralytics lacks YOLO11 C3k2 support: {exc}")
    else:
        report.ok("Ultralytics provides YOLO11 C3k2")

    try:
        import torch

        cuda_build = torch.version.cuda
        if cuda_build != "12.1":
            report.error(f"PyTorch CUDA build is {cuda_build}; expected 12.1")
        else:
            report.ok("PyTorch CUDA build is 12.1")
        if torch.cuda.is_available():
            report.ok(f"CUDA device available: {torch.cuda.get_device_name(0)}")
        elif allow_no_cuda:
            report.warn("CUDA is unavailable; dependency checks can pass, but full inference cannot run")
        else:
            report.error("CUDA is unavailable; HMR2 and motion inference require an NVIDIA GPU")
    except Exception as exc:
        report.error(f"could not inspect PyTorch CUDA support: {exc}")

    try:
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError) as exc:
        report.error(f"ffmpeg executable is unavailable: {exc}")
    else:
        report.ok("ffmpeg executable")


def _resolve_mesh_paths(xml_path: Path) -> Iterable[Path]:
    root = ET.parse(xml_path).getroot()
    if root.tag == "robot":
        compiler = root.find("mujoco/compiler")
        mesh_dir = Path(compiler.attrib.get("meshdir", "")) if compiler is not None else Path()
        for mesh in root.findall(".//mesh"):
            filename = mesh.attrib.get("filename")
            if not filename or filename.startswith("package://"):
                continue
            path = Path(filename)
            yield path if path.is_absolute() else xml_path.parent / mesh_dir / path
        return

    if root.tag == "mujoco":
        compiler = root.find("compiler")
        mesh_dir = Path(compiler.attrib.get("meshdir", "")) if compiler is not None else Path()
        assets = root.find("asset")
        if assets is None:
            return
        for mesh in assets.findall("mesh"):
            filename = mesh.attrib.get("file")
            if not filename:
                continue
            path = Path(filename)
            yield path if path.is_absolute() else xml_path.parent / mesh_dir / path


def check_robot_assets(report: CheckReport, project_root: Path) -> None:
    try:
        from lib.robot.robot_kinematics import RobotKinematicsModel
        from lib.robot.robot_spec import load_robot_spec
    except Exception as exc:
        report.error(f"cannot import robot validation modules: {exc}")
        return

    robot_root = project_root / "assets/robot"
    if robot_root.is_symlink():
        report.error("assets/robot is a symbolic link")
        return
    symlinks = [path for path in robot_root.rglob("*") if path.is_symlink()] if robot_root.exists() else []
    if symlinks:
        report.error(f"robot bundle contains symbolic links, first: {symlinks[0]}")
        return

    for robot_name, relative_spec in ROBOT_SPEC_PATHS.items():
        spec_path = project_root / relative_spec
        try:
            spec = load_robot_spec(str(spec_path), project_root=str(project_root))
            xml_path = Path(spec.xml_path)
            if not xml_path.is_file() or xml_path.is_symlink():
                raise FileNotFoundError(f"robot description is missing or is a symlink: {xml_path}")
            missing_meshes = [path for path in _resolve_mesh_paths(xml_path) if not path.is_file()]
            if missing_meshes:
                raise FileNotFoundError(f"referenced mesh is missing: {missing_meshes[0]}")
            for item in spec.foot_grounding_meshes.values():
                mesh_path = Path(item["mesh_path"])
                if not mesh_path.is_file():
                    raise FileNotFoundError(f"foot grounding mesh is missing: {mesh_path}")
            model = RobotKinematicsModel(
                str(xml_path),
                device="cpu",
                model_to_xml_dof=spec.model_to_xml_dof,
                neutral_dof=spec.neutral_dof,
            )
            parsed_dof = int(model._num_dof)
            expected_xml_dof = int(spec.xml_dof or spec.dof)
            if parsed_dof != expected_xml_dof:
                raise ValueError(f"parsed {parsed_dof} DoFs, expected {expected_xml_dof}")
        except Exception as exc:
            report.error(f"robot {robot_name}: {exc}")
        else:
            report.ok(f"robot {robot_name}: description, meshes, and {parsed_dof} DoFs")


def check_model_assets(report: CheckReport, project_root: Path, deep: bool) -> None:
    asset_errors: Dict[str, str] = {}
    for asset in ASSETS:
        error = validate_asset_file(project_root / asset.destination, asset)
        if error:
            asset_errors[asset.destination] = error
            report.error(f"{asset.destination}: {error}")
        else:
            verification = "size and SHA256" if asset.sha256 else "expected size"
            report.ok(f"{asset.destination}: {verification}")

    onnx_path = project_root / "stream_infer/assets/yolo11s.onnx"
    if onnx_path.is_file():
        report.ok("user-provided stream_infer/assets/yolo11s.onnx")
    else:
        report.ok("streaming detector is optional; see stream_infer/README.md to prepare it")

    check_robot_assets(report, project_root)
    if not deep:
        return

    yolo_path = project_root / "assets/yolo/yolov8x.pt"
    if "assets/yolo/yolov8x.pt" not in asset_errors:
        try:
            from ultralytics import YOLO

            model = YOLO(str(yolo_path))
            if str(model.names.get(0, "")).lower() != "person":
                raise ValueError(f"class 0 is {model.names.get(0)!r}, expected 'person'")
        except Exception as exc:
            report.error(f"could not load YOLOv8x checkpoint: {exc}")
        else:
            report.ok("YOLOv8x checkpoint loads and class 0 is person")

    if onnx_path.is_file():
        try:
            import onnxruntime as ort

            session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            if not session.get_inputs() or not session.get_outputs():
                raise ValueError("ONNX graph has no inputs or outputs")
        except Exception as exc:
            report.error(f"could not load streaming YOLO11 ONNX detector: {exc}")
        else:
            report.ok("streaming YOLO11 ONNX detector loads with ONNX Runtime CPU")

    checkpoint_path = project_root / "assets/checkpoints/rgb2robo_multirobot_clean.pth"
    if "assets/checkpoints/rgb2robo_multirobot_clean.pth" not in asset_errors:
        try:
            requested = list(ROBOT_SPEC_PATHS)
            import torch

            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if not isinstance(checkpoint, dict):
                raise ValueError(f"expected a checkpoint dictionary, got {type(checkpoint).__name__}")
            state = checkpoint.get("model_state_dict", checkpoint)
            if not isinstance(state, dict) or not state:
                raise ValueError("model_state_dict is missing or empty")
            heads = set((checkpoint.get("robot_heads") or {}).keys())
            heads.update(
                str(key).split(".")[1]
                for key in state
                if str(key).startswith("robot_heads.") and len(str(key).split(".")) > 2
            )
            missing = sorted(set(requested) - heads)
            if missing:
                raise ValueError(f"missing robot heads: {', '.join(missing)}")
            has_contact = bool(checkpoint.get("contact_head_state_dict")) or any(
                str(key).startswith("contact_predictor.") for key in state
            )
            if not has_contact:
                raise ValueError("contact predictor weights are missing")
            from postprocess.contact_infer import load_contact_head
            from scripts.run_multirobot_pipeline import checkpoint_robot_heads, load_unified_model

            head_metadata = checkpoint_robot_heads(str(checkpoint_path), requested=requested)
            model = load_unified_model(
                str(project_root / "config/inference.yaml"),
                str(checkpoint_path),
                torch.device("cpu"),
                head_metadata,
            )
            contact_model = load_contact_head(str(checkpoint_path), torch.device("cpu"))

            window_length = int(model.frontend.cross_window_config.get("window_length", 50))
            context_span = int(model.frontend.cross_window_config.get("context_span", 8))
            input_dim = int(model.frontend.input_feature_dim)
            visual = torch.zeros(1, window_length, input_dim)
            context = {"img_kv": torch.zeros(1, context_span * 2, input_dim)}
            context_mask = torch.ones(1, context_span * 2, dtype=torch.bool)
            valid_mask = torch.ones(1, window_length, dtype=torch.bool)
            with torch.no_grad():
                motion_feature = model.extract_motion_feature(
                    visual,
                    context_kv=context,
                    window_mask=context_mask,
                    valid_mask=valid_mask,
                )
                predictions = {
                    name: model.predict_from_feature(
                        motion_feature,
                        1,
                        window_length,
                        name,
                        valid_mask=valid_mask,
                    )["motion"]
                    for name in requested
                }
                contact_logits = contact_model(
                    motion_feature.view(1, window_length, -1),
                    visual,
                    valid_mask=valid_mask,
                )
            invalid = [name for name, value in predictions.items() if not torch.isfinite(value).all()]
            if invalid or not torch.isfinite(contact_logits).all():
                raise ValueError(f"non-finite smoke-test output for: {', '.join(invalid) or 'contact head'}")
            if tuple(contact_logits.shape) != (1, window_length, 2):
                raise ValueError(f"unexpected contact output shape: {tuple(contact_logits.shape)}")
        except Exception as exc:
            report.error(f"invalid RGB2Robo checkpoint: {exc}")
        else:
            report.ok("RGB2Robo checkpoint runs all robot heads and the contact head")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--allow-no-cuda", action="store_true")
    parser.add_argument("--skip-assets", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Skip model deserialization and ONNX Runtime checks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    report = CheckReport()
    check_environment(report, allow_no_cuda=args.allow_no_cuda)
    if not args.skip_assets:
        check_model_assets(report, project_root, deep=not args.quick)

    print(f"\nSummary: {len(report.errors)} error(s), {len(report.warnings)} warning(s)")
    if report.errors:
        raise SystemExit(1)
    if report.warnings:
        print("Inference checks completed; review the warnings above.")
    else:
        print("RGB2Robo is ready for inference.")


if __name__ == "__main__":
    main()
