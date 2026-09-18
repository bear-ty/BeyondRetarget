# Third-Party Notices

## HMR2 Encoder

`lib/vendor/hmr2/` includes unmodified ViT and Transformer modules from
[4D-Humans](https://github.com/shubham-goel/4D-Humans).

Copyright (c) 2023 UC Regents, Shubham Goel. The MIT license is retained in
`lib/vendor/hmr2/LICENSE`.

The ViT implementation retains its OpenMMLab copyright notice. The Apache 2.0
license from [ViTPose](https://github.com/ViTAE-Transformer/ViTPose) is retained
in `lib/vendor/hmr2/OPENMMLAB_LICENSE`.

HMR2 weights and other third-party assets retain their upstream terms.

## YOLO

BeyondRetarget uses [Ultralytics YOLO](https://github.com/ultralytics/ultralytics):
YOLOv8 for offline person tracking and YOLO11 for streaming detection.
We thank the Ultralytics authors and contributors for their models and tools.

Ultralytics software and pretrained weights retain their upstream terms; see
[AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/v8.3.0/LICENSE) and
[Ultralytics licensing](https://www.ultralytics.com/license). 

## Robot Assets

The eight robot descriptions and meshes distributed in `robot.zip` retain
their directory-level licenses. Each directory includes `LICENSE` and
`SOURCE.md`; T1 also includes `NOTICE`.

| Robot resources | Upstream terms |
| --- | --- |
| Unitree G1, R1, H1; Tienkung | BSD-3-Clause |
| Fourier GR1T1, GR2 | GPL-3.0 |
| Booster T1 | Apache-2.0, with NOTICE |
| Atlas V4 | Roboschool MIT text and MuJoCo model attribution |

These resources are not relicensed under this project's CC BY-NC 4.0.
