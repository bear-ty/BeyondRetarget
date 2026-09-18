# Motion Post-processing

Post-processing uses predicted left/right foot contacts to smooth motion,
correct root position, and refine leg poses with CCD inverse kinematics.

Run through the [offline pipeline](../README.md#offline-inference)
with `--phase all`, or use `--phase postprocess` for existing predictions.
Results are saved to `outputs/inference/postprocess/`. For all options:

```bash
python scripts/run_multirobot_pipeline.py --help
```

For Python integration, use `final_robot_postprocess.py`.
`postprocess_robot_motion.py` and `robot_ik_postprocess.py` provide its internal
filtering and IK operations.
