# Project Context: shm-vibration-tracking (Multi-Camera AprilTag 3D Displacement & Vibration Tracking)

## Goal
Build a reproducible research codebase and publish Paper 2: a novel offline algorithmic framework for
high-fidelity 3D vibration/displacement tracking from consumer cameras by fusing AprilTag pose with
KLT / optical-flow recovery and self-validation.

## System Overview
- 3x Sony RX10 IV cameras, target 60 FPS, captured via AverMedia capture cards.
- ROS Noetic used for capture + rosbag storage; heavy processing must run offline.
- Known issues in realtime: CPU overload → dropped frames → sync errors; motion blur → tag detection failures.
- Offline workflow: Bag → (optional export PNG) → detection → world transform → sync/resample → fusion/smoothing → metrics/plots.

## Repository Layout (authoritative)
- `ros/` : recording / playback launch files and thin ROS nodes (no heavy processing).
- `offline/src/shmtrack/` : core library (IO, detection, sync, fusion, analysis).
- `offline/scripts/` : stable CLIs (the “capability interface”).
- `experiments/` : YAML configs + runner for full pipeline reproducibility.
- `results/<run>/...` : all outputs (never committed).
- `data/` : calibration + small versionable config; large rosbags live outside git.

## Coordinate Frames / Data
- Intrinsics per camera (K, distortion).
- Extrinsics define camera frame -> world/marker frame (YAML in `data/calib/`).
- Output of interest: 6DoF pose time series and derived displacement:
  ΔT(t) = T(t) * inv(T0) (and axis-specific displacement in world coordinates).

## Timebase Policy (critical)
- Preserve both:
  - `t_bag_sec`: rosbag time.
  - `t_hdr_sec`: msg.header.stamp time (when available).
- For analysis, default timebase is header-based per-camera normalized by that camera’s first header stamp:
  `t_sec = t_hdr_sec - t_hdr0_cam`.
- Never clamp/shift time silently except for documented normalization.
- Multi-camera alignment is offline (drift/offset estimation + resampling to master grid).

## Canonical Schema Policy
- Internally use meaningful canonical columns (v2), e.g. `t_sec`, `x_m`, `decision_margin`, `tag_area_px2`.
- Provide legacy CSV export compatibility (old scripts/paper figures).
- Do not invent new column names ad hoc; update `offline/src/shmtrack/io/schema.py`.

## Paper 2 Novelty Targets (distinct from Paper 1 / IEEE Access)
Paper 1: system + AprilTag-only displacement + ROS integration.
Paper 2: algorithmic novelty, offline robustness:
1) AprilTag + KLT corner tracking fallback (pose recovery during dropout/blur)
2) Dense optical flow ROI motion estimates
3) Motion Consistency Index (MCI) for outlier rejection / self-validation
4) Confidence-weighted fusion + smoothing
5) Demonstrate improved pose continuity and frequency-domain stability (FFT/FDD-like metrics)

## Algorithms to Implement
- AprilTag primary pose: detect corners, solvePnP, reprojection error.
- KLT fallback: LK tracking of corners when tag lost, re-run solvePnP, log quality.
- Dense flow ROI: flow magnitude as motion-energy proxy.
- MCI: compare pose-derived velocity vs flow-derived proxy; reject/downweight inconsistent updates.
- Fusion: translation weighted; rotation via quaternion SLERP; RTS/Kalman smoothing offline.
- Offline sync: drift/offset estimation + resampling/interpolation to common grid.

## Coding Rules
- Python 3.8+.
- Keep modules separated and testable.
- Deterministic outputs; log per-frame quality metrics.
- No rosbags/exports/results committed to git.

## Evaluation Requirements (Paper Figures)
Compare 3 pipelines:
1) AprilTag-only
2) AprilTag + KLT fallback
3) Full fusion + MCI + motion-aware processing

Metrics:
- Pose continuity (% usable frames)
- Outlier/rejection rate
- Sync residual (ms) after alignment
- FFT peak stability and SNR improvement
- Sensitivity to simulated dropout/blur

## Assistant Behavior
- Prefer robust, testable implementations and small incremental commits.
- Preserve Paper 1 parity unless explicitly changing as Paper 2 contribution.
- When proposing methods: give equations + pseudocode + what to plot.
