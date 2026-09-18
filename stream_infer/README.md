# Streaming Inference

Run G1 motion inference from a video or camera using the BeyondRetarget
checkpoint and contact head. Complete [installation](../README.md#installation),
activate `BeyondRetarget`, and run commands from the project root.

HMR2 and motion inference require CUDA. Streaming detectors are not included
in the repository or downloaded by setup.

Download `yolo11s.pt` from the project's
[Google Drive folder](https://drive.google.com/drive/folders/12ySWLbVhF8WEGyLR0Ll09DmZm8rZE0Tg)
and place it under `stream_infer/assets/` (create this directory first).
Export the ONNX detector with dynamic batch support:

```bash
yolo export model=stream_infer/assets/yolo11s.pt format=onnx imgsz=640 dynamic=True simplify=False opset=17 device=cpu
```

This creates `stream_infer/assets/yolo11s.onnx`. ONNX inference uses ONNX Runtime
and falls back to CPU FP32 when a CUDA provider is unavailable; TensorRT is
not required. Alternatively, use the PyTorch detector directly with
`--yolo_ckpt stream_infer/assets/yolo11s.pt`.
Optional TensorRT engines require a separate compatible TensorRT installation
and local export.

The live pipeline defaults to detector batch 2, feature batch 2, and motion
stride 4. The dynamic ONNX model also accepts warmup and partial batches
without padding.

Run fixed-lag streaming on a file:

```bash
python stream_infer/run_streaming_video.py \
  --video_path path/to/video.mp4 \
  --hmr2_ckpt assets/hmr2/epoch=10-step=25000.ckpt
```

Use `--overwrite` to replace existing file-streaming outputs, or choose a new
directory with `--output_dir`.

Run a live camera:

```bash
python stream_infer/run_live_camera.py \
  --input_source camera \
  --camera_index 0
```

For a local video, use `--input_source video --video_path path/to/video.mp4`.
Add `--enable_groot_zmq` to publish actions to GR00T through ZMQ, or
`--enable_mujoco_viewer` to display them in MuJoCo.
The standalone `live_mujoco_viewer.py` replays a saved action stream, and
`groot_zmq_publisher.py` publishes a saved action stream. Outputs default to
`outputs/streaming` and `outputs/live_camera`.
