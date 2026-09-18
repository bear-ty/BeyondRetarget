#!/usr/bin/env python3
"""Download and install the external assets required by RGB2Robo."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DRIVE_FOLDER = "https://drive.google.com/drive/folders/12ySWLbVhF8WEGyLR0Ll09DmZm8rZE0Tg"
EXPECTED_ROBOT_DIRS = {
    "atlas_v4",
    "gr1t1",
    "gr2v3_8_7_dummy_hand",
    "h1_with_hand",
    "t1_serial",
    "tienkung",
    "unitree_g1",
    "unitree_r1",
}


@dataclass(frozen=True)
class Asset:
    remote_name: str
    destination: str
    size: int
    sha256: Optional[str] = None
    file_id: Optional[str] = None


ASSETS = (
    Asset(
        remote_name="epoch=10-step=25000.ckpt",
        destination="assets/hmr2/epoch=10-step=25000.ckpt",
        size=2_709_494_041,
        sha256="2dcf79638109781d1ae5f5c44fee5f55bc83291c210653feead9b7f04fa6f20e",
        file_id="1jCByDYWG8gYpfoK32GK7BhXwOvCWmO6f",
    ),
    Asset(
        remote_name="rgb2robo_multirobot.pth",
        destination="assets/checkpoints/rgb2robo_multirobot_clean.pth",
        size=415_291_282,
        sha256="6a145fe08c862eb7203d452fd85f18c5d9bd0846ef476a2d56bcfb90f2fa76f9",
        file_id="1tTm_Hi0f_SLUhf-LEHAyXpB2HlNmr_U1",
    ),
    Asset(
        remote_name="robot.zip",
        destination="assets/robot.zip",
        size=92_183_004,
        sha256="9091c222d0026fd38ce4e8fda8b44515dc74a10f0c9b6ff55ebaed6f5d39c5ec",
        file_id=None,
    ),
    Asset(
        remote_name="yolov8x.pt",
        destination="assets/yolo/yolov8x.pt",
        size=136_890_692,
        sha256="3df4ada6b4dad6d657868f2fdf7faecfb34dcfccf3a25c4b82079064718524c8",
        file_id="1EFAUwOgSmHwkHAz3-6wMiA9sI-jhGcCg",
    ),
)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_asset_file(path: Path, asset: Asset) -> Optional[str]:
    if path.is_symlink():
        return "is a symbolic link; downloaded release assets must be regular files"
    if not path.is_file():
        return "is missing"
    actual_size = path.stat().st_size
    if actual_size != asset.size:
        return f"has size {actual_size}, expected {asset.size}"
    if asset.sha256:
        actual_hash = sha256_file(path)
        if actual_hash != asset.sha256:
            return f"has SHA256 {actual_hash}, expected {asset.sha256}"
    return None


def _decode_drive_name(value: str) -> str:
    return re.sub(
        r"\\u([0-9a-fA-F]{4})",
        lambda match: chr(int(match.group(1), 16)),
        value,
    )


def parse_drive_folder_html(html: str) -> Dict[str, Dict[str, object]]:
    match = re.search(r"window\['_DRIVE_ivd'\]\s*=\s*'(.*?)';if", html, re.DOTALL)
    if not match:
        raise RuntimeError("Google Drive folder metadata was not found in the public page")
    json_text = re.sub(
        r"\\x([0-9a-fA-F]{2})",
        lambda item: chr(int(item.group(1), 16)),
        match.group(1),
    )
    try:
        rows = json.loads(json_text)[0]
    except (IndexError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Google Drive folder metadata has an unsupported format") from exc

    entries: Dict[str, Dict[str, object]] = {}
    for row in rows:
        if not isinstance(row, list) or len(row) <= 13:
            continue
        file_id = row[0]
        name = row[2]
        size = row[13]
        if not isinstance(file_id, str) or not isinstance(name, str):
            continue
        entries[_decode_drive_name(name)] = {
            "id": file_id,
            "size": int(size) if isinstance(size, (int, float)) else None,
        }
    return entries


def fetch_drive_folder(folder_url: str) -> Dict[str, Dict[str, object]]:
    request = urllib.request.Request(
        folder_url,
        headers={"User-Agent": "Mozilla/5.0 RGB2Robo setup"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            html = response.read().decode("utf-8", errors="replace")
    except Exception as exc:
        raise RuntimeError(f"could not read public Google Drive folder: {exc}") from exc
    return parse_drive_folder_html(html)


def resolve_assets(folder_url: str, yolov8x_file_id: Optional[str]) -> Iterable[Asset]:
    entries: Dict[str, Dict[str, object]] = {}
    fetch_error: Optional[Exception] = None
    try:
        entries = fetch_drive_folder(folder_url)
    except Exception as exc:
        fetch_error = exc

    resolved = []
    for asset in ASSETS:
        entry = entries.get(asset.remote_name)
        file_id = asset.file_id
        uses_yolov8x_override = asset.remote_name == "yolov8x.pt" and bool(yolov8x_file_id)
        if uses_yolov8x_override:
            file_id = yolov8x_file_id
        elif entry:
            file_id = str(entry["id"])

        if entry and not uses_yolov8x_override and entry.get("size") != asset.size:
            raise RuntimeError(
                f"Google Drive file {asset.remote_name!r} has size {entry.get('size')}, "
                f"expected {asset.size}. Check that the correct file was uploaded."
            )
        if not file_id:
            listed = ", ".join(sorted(entries)) or "none"
            detail = f" Folder lookup failed: {fetch_error}" if fetch_error else ""
            raise RuntimeError(
                f"Required file {asset.remote_name!r} is not publicly visible in the Drive folder. "
                f"Visible files: {listed}.{detail} Make the file public through the shared folder, "
                "or pass --yolov8x-file-id with its public file ID."
            )
        resolved.append(replace(asset, file_id=file_id))
    return resolved


def _parse_content_range(value: Optional[str]) -> tuple[int, int, int]:
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", value or "")
    if not match:
        raise RuntimeError(f"invalid Google Drive Content-Range header: {value!r}")
    return tuple(int(item) for item in match.groups())


def _request_drive_range(file_id: str, start: int, end: int, expected_size: int) -> bytes:
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 RGB2Robo setup",
            "Range": f"bytes={start}-{end}",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        status = getattr(response, "status", response.getcode())
        if status != 206:
            raise RuntimeError(f"Google Drive ignored byte range {start}-{end}: HTTP {status}")
        actual_start, actual_end, total = _parse_content_range(response.headers.get("Content-Range"))
        if (actual_start, actual_end, total) != (start, end, expected_size):
            raise RuntimeError(
                f"unexpected Google Drive range {(actual_start, actual_end, total)}, "
                f"expected {(start, end, expected_size)}"
            )
        data = response.read()
    expected_length = end - start + 1
    if len(data) != expected_length:
        raise RuntimeError(f"short Google Drive range: received {len(data)}, expected {expected_length}")
    return data


def download_drive_file(file_id: str, output: Path, expected_size: int, fingerprint: str) -> None:
    segment_size = 16 * 1024 * 1024
    metadata_path = output.with_name(output.name + ".meta.json")
    expected_metadata = {"file_id": file_id, "size": expected_size, "fingerprint": fingerprint}
    resume_valid = False
    if output.is_file() and not output.is_symlink() and metadata_path.is_file():
        try:
            resume_valid = json.loads(metadata_path.read_text(encoding="utf-8")) == expected_metadata
        except (OSError, json.JSONDecodeError):
            resume_valid = False
        resume_valid = resume_valid and output.stat().st_size <= expected_size
    if not resume_valid:
        if output.exists() or output.is_symlink():
            output.unlink()
        metadata_path.write_text(json.dumps(expected_metadata, sort_keys=True), encoding="utf-8")

    start = output.stat().st_size if output.exists() else 0
    while start < expected_size:
        end = min(start + segment_size, expected_size) - 1
        last_error: Optional[Exception] = None
        for attempt in range(5):
            try:
                data = _request_drive_range(file_id, start, end, expected_size)
                break
            except Exception as exc:
                last_error = exc
                if attempt == 4:
                    raise RuntimeError(f"failed to download byte range {start}-{end}: {exc}") from exc
                time.sleep(min(2**attempt, 8))
        else:
            raise RuntimeError(f"failed to download byte range {start}-{end}: {last_error}")
        with output.open("ab") as handle:
            handle.write(data)
        start = end + 1
        print(f"[asset] downloaded {start}/{expected_size} bytes", flush=True)
    metadata_path.unlink(missing_ok=True)


def download_asset(asset: Asset, project_root: Path, force: bool) -> None:
    destination = project_root / asset.destination
    existing_error = validate_asset_file(destination, asset)
    if existing_error is None and not force:
        print(f"[asset] ready: {asset.destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    part_path = destination.with_name(destination.name + ".part")
    if force and (part_path.exists() or part_path.is_symlink()):
        part_path.unlink()
    print(f"[asset] downloading {asset.remote_name} -> {asset.destination}")
    fingerprint = asset.sha256 or f"size:{asset.size}"
    download_drive_file(str(asset.file_id), part_path, asset.size, fingerprint)
    error = validate_asset_file(part_path, asset)
    if error:
        if part_path.exists() or part_path.is_symlink():
            part_path.unlink()
        part_path.with_name(part_path.name + ".meta.json").unlink(missing_ok=True)
        raise RuntimeError(f"downloaded {asset.remote_name} {error}")
    os.replace(part_path, destination)


def _safe_zip_members(archive: zipfile.ZipFile) -> Iterable[zipfile.ZipInfo]:
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        file_type = (info.external_attr >> 16) & 0o170000
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError(f"unsafe path in robot.zip: {info.filename}")
        if not path.parts or path.parts[0] != "robot":
            raise RuntimeError(f"robot.zip entry is outside robot/: {info.filename}")
        if file_type == stat.S_IFLNK:
            raise RuntimeError(f"symbolic link is not allowed in robot.zip: {info.filename}")
        yield info


def install_robot_bundle(project_root: Path, force: bool = False) -> None:
    archive_path = project_root / "assets/robot.zip"
    robot_root = project_root / "assets/robot"
    expected = {robot_root / name for name in EXPECTED_ROBOT_DIRS}
    with zipfile.ZipFile(archive_path) as archive:
        members = list(_safe_zip_members(archive))
        top_dirs = {PurePosixPath(item.filename).parts[1] for item in members if len(PurePosixPath(item.filename).parts) > 1}
        missing = sorted(EXPECTED_ROBOT_DIRS - top_dirs)
        forbidden = sorted({"bluewhale", "penguin_15dof", "romeo"} & top_dirs)
        if missing:
            raise RuntimeError(f"robot.zip is missing robot directories: {', '.join(missing)}")
        if forbidden:
            raise RuntimeError(f"robot.zip contains excluded robot directories: {', '.join(forbidden)}")

        files_match_archive = all(
            info.is_dir()
            or (
                (project_root / "assets" / PurePosixPath(info.filename)).is_file()
                and not (project_root / "assets" / PurePosixPath(info.filename)).is_symlink()
                and (project_root / "assets" / PurePosixPath(info.filename)).stat().st_size == info.file_size
            )
            for info in members
        )
        tree_ready = (
            not robot_root.is_symlink()
            and all(path.is_dir() and not path.is_symlink() for path in expected)
            and not any(path.is_symlink() for path in robot_root.rglob("*"))
            and files_match_archive
        )
        if tree_ready and not force:
            print("[asset] ready: assets/robot/")
            return

        assets_root = project_root / "assets"
        with tempfile.TemporaryDirectory(prefix="robot_extract_", dir=assets_root) as temp_dir:
            temp_root = Path(temp_dir)
            archive.extractall(temp_root, members=members)
            extracted = temp_root / "robot"
            if any(path.is_symlink() for path in extracted.rglob("*")):
                raise RuntimeError("extracted robot bundle contains symbolic links")
            if robot_root.is_symlink():
                robot_root.unlink()
            shutil.copytree(extracted, robot_root, dirs_exist_ok=True)
    print("[asset] installed: assets/robot/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--folder-url", default=DEFAULT_DRIVE_FOLDER)
    parser.add_argument("--yolov8x-file-id", default=os.environ.get("RGB2ROBO_YOLOV8X_FILE_ID"))
    parser.add_argument("--force", action="store_true", help="Download valid existing files again.")
    parser.add_argument("--check-only", action="store_true", help="Check local assets without network access.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    if args.check_only:
        errors = []
        for asset in ASSETS:
            error = validate_asset_file(project_root / asset.destination, asset)
            if error:
                errors.append(f"{asset.destination}: {error}")
            else:
                print(f"[asset] ready: {asset.destination}")
        if errors:
            raise SystemExit("Asset check failed:\n  - " + "\n  - ".join(errors))
        install_robot_bundle(project_root)
        return

    assets = list(resolve_assets(args.folder_url, args.yolov8x_file_id))
    for asset in assets:
        download_asset(asset, project_root, force=args.force)
    install_robot_bundle(project_root, force=args.force)
    print("All downloadable RGB2Robo assets are installed.")


if __name__ == "__main__":
    main()
