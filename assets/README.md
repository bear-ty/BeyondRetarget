# External Assets

Run [setup](../README.md#installation) to download the checkpoints and robot
bundle to the paths below.

| Asset | Required destination |
| --- | --- |
| BeyondRetarget checkpoint | `assets/checkpoints/rgb2robo_multirobot_clean.pth` |
| HMR2 feature-extractor checkpoint | `assets/hmr2/epoch=10-step=25000.ckpt` |
| YOLOv8x person-tracker checkpoint | `assets/yolo/yolov8x.pt` |
| Supported robot URDF/MJCF and meshes | Extract `robot.zip` as `assets/robot/` |

Streaming detectors are prepared separately; setup does not download them.
See [streaming setup](../stream_infer/README.md) to export `yolo11s.onnx`,
or supply `yolo11s.pt` with `--yolo_ckpt`.

The archive must create this layout:

```text
assets/robot/
  atlas_v4/
  gr1t1/
  gr2v3_8_7_dummy_hand/
  h1_with_hand/
  t1_serial/
  tienkung/
  unitree_g1/
  unitree_r1/
```

Keep the robot directory structure intact: descriptions reference the meshes
used by kinematics, post-processing, and visualization.

Each robot directory includes its upstream `LICENSE` and `SOURCE.md`.
T1 also includes `NOTICE`; Fourier GR1T1 and GR2 resources remain GPL-3.0. Some robots' URDF files have been adjusted.
See [third-party notices](../THIRD_PARTY_NOTICES.md#robot-assets).
