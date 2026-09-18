#!/usr/bin/env python3
"""Check prepared inference features and optional boxes against source frames."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.util.gen_image_feature import list_rgb_frames
from lib.util.inference_inputs import load_bbox, load_visual_features
from lib.util.video_io import video_info


def discover_sequences(root: Path) -> list[Path]:
    if not root.is_dir():
        raise NotADirectoryError(root)
    sequences, visited = [], set()
    def raise_walk_error(error):
        raise error
    for directory, children, files in os.walk(root, followlinks=True, onerror=raise_walk_error):
        path = Path(directory)
        if path.resolve() in visited:
            children[:] = []
            continue
        visited.add(path.resolve())
        children[:] = sorted(name for name in children if not name.startswith("."))
        has_video = any(Path(name).suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"} for name in files)
        if {"vit_features.pt", "bbox.npy"}.intersection(files) or "color" in children or has_video or not children:
            sequences.append(path)
            children[:] = []
    return sorted(sequences)


def validate_sequence(path: Path) -> list[str]:
    errors = []
    counts = {}
    if (path / "color").is_dir():
        try:
            counts["images"] = len(list_rgb_frames(path / "color"))
        except Exception as exc:
            errors.append(str(exc))
    else:
        videos = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"} and p.name != "_tracker_input.mp4")
        source = path / "1.mp4" if (path / "1.mp4").is_file() else (videos[0] if len(videos) == 1 else None)
        if source is not None:
            try:
                counts["video"] = video_info(source)[0]
            except Exception as exc:
                errors.append(f"cannot read {source.name}: {exc}")
        elif len(videos) > 1:
            errors.append("ambiguous source videos; use one video per sequence directory")
    for filename, loader in (("vit_features.pt", load_visual_features), ("bbox.npy", load_bbox)):
        if filename == "bbox.npy" and not (path / filename).exists():
            continue
        try:
            counts[filename] = len(loader(path / filename))
        except Exception as exc:
            errors.append(f"{filename}: {exc}")
    if counts and (min(counts.values()) <= 0 or len(set(counts.values())) != 1):
        errors.append(f"frame-count mismatch or empty sequence: {counts}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    sequences = discover_sequences(args.root.expanduser().resolve())
    if not sequences:
        parser.error(f"No sequences found under {args.root}")
    failures = {}
    for path in sequences:
        errors = validate_sequence(path)
        if errors:
            failures[str(path)] = errors
    print(f"validated={len(sequences)} invalid={len(failures)}")
    for path, errors in failures.items():
        print(f"{path}: {'; '.join(errors)}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
