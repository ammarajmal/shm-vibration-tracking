# Calibration

Accurate intrinsics and extrinsics are fundamental for 3D recovery.

## Intrinsic Calibration
*   **Model**: Pinhole + Distortion (Rad-Tan).
*   **Storage**: Inside `camera_info` topics in the ROS bag, or overridden via YAML in `data/calib/`.
*   **Format**: Standard OpenCV `K` (3x3 matrix) and `D` (distortion coefficients).

## Extrinsic Calibration (Camera -> World)
We define the **World Frame** typically attached to a reference static tag or aligned with gravity.

### 1. Tag-Based Extrinsics
For simple setups, we assume the AprilTag is at World Origin $(0,0,0)$.
*   The camera pose $T_{cw}$ (World to Camera) is simply the inverse of the detected tag pose $T_{tag}$.

### 2. Multi-Cam Bundle Adjustment (Optional)
For high-precision fusion, we fix the relative transforms between cameras $T_{c1\_c2}, T_{c1\_c3}$.
*   computed by viewing a common static target (checkerboard or tag array) before the experiment.
*   These transforms are stored in `data/calib/extrinsics.yaml`.
