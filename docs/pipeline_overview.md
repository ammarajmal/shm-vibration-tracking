# Pipeline Overview

This document describes the end-to-end data processing pipeline for the **shm-vibration-tracking** framework. The pipeline is designed to be fully **offline**, prioritizing traceability, robustness, and high-fidelity signal recovery over real-time performance.

## 🔄 High-Level Workflow

The pipeline consists of four main stages:

1.  **Data Acquisition (ROS)**
    *   Capture raw video streams from 3x Sony RX10 IV cameras.
    *   Store data in ROS bags (`.bag`).
    *   *No processing* is done here to avoid CPU load and frame drops.

2.  **Extraction & Normalization**
    *   **Input**: `raw_data.bag`
    *   **Process**: `bag_export.py` reads relevant topics.
    *   **Output**: 
        *   PNG sequence for every frame (lossless).
        *   `frames.csv`: Normalized per-camera timestamps (`t_sec` starting at 0.0).

3.  **Vision & Detection (Per-Camera)**
    *   **Input**: PNG frames + `camera_info.yaml`.
    *   **Process**:
        *   **Baseline**: `detect_baseline.py` runs standard AprilTag detection.
        *   **Feature Tracking**: `klt_fallback.py` tracks corners when tags are lost (Paper 2).
        *   **Flow ROI**: `flow_roi.py` computes dense optical flow variance.
    *   **Output**: `detect/baseline/<cam>_poses.csv` (raw, unaligned poses).

4.  **Multi-Camera Fusion & Analysis**
    *   **Input**: Per-camera detection CSVs.
    *   **Process**:
        *   **Synchronization**: Align streams to a common master time grid.
        *   **Self-Validation**: Calculate Motion Consistency Index (MCI).
        *   **Fusion**: Weighted average of poses from all visible cameras.
        *   **Smoothing**: RTS (Rauch-Tung-Striebel) smoothing.
    *   **Output**: `results/<run>/fused_trajectory.csv`.

## 📂 Directory Structure flow

```
raw_data/
  └── experiment_1.bag

results/
  └── experiment_1/
      ├── export/
      │   ├── sony_cam1/
      │   │   ├── frames/ (000001.png ...)
      │   │   ├── frames.csv
      │   │   └── camera_info.yaml
      │   └── ...
      ├── detect/
      │   ├── baseline/
      │   │   ├── sony_cam1_poses.csv
      │   │   └── ...
      │   └── fused/
      │       ├── fused_trajectory.csv
      │       └── plots/
```
