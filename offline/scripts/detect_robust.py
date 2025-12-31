#!/usr/bin/env python3
"""
Robust Detection Script (Paper 2).

Integrates AprilTag detection with KLT fallback tracking and Optical Flow analysis.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, List, Set

import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm

from shmtrack.io.schema import (
    T, CAM, TAG, X, Y, Z, QX, QY, QZ, QW,
    DECISION_MARGIN, HAMMING, TAG_AREA,
    DETECTED, SOURCE, REPROJ_ERR,
    KLT_RMSE, FLOW_MAG,
    ensure_columns, export_legacy_detect_csv
)
from shmtrack.vision.apriltag_pupil import detect_tags, pick_best_single_tag
from shmtrack.vision.pnp_pose import solve_pnp_from_corners
from shmtrack.io.rosbag_reader import CameraInfoData
from shmtrack.vision.klt_fallback import track_corners_lk
from shmtrack.vision.flow_roi import compute_flow_roi


def load_camera_info_yaml(caminfo_path: Path) -> CameraInfoData:
    import yaml
    d = yaml.safe_load(caminfo_path.read_text())
    return CameraInfoData(
        K=np.array(d["K"], dtype=float),
        D=np.array(d["D"], dtype=float),
        width=int(d.get("width", 0)),
        height=int(d.get("height", 0)),
        frame_id=str(d.get("frame_id", "")),
    )

def process_frames(
    df_frames: pd.DataFrame,
    caminfo: CameraInfoData,
    cam_id: str,
    tag_id: Optional[int],
    det_params: dict,
    family: str,
    tag_size_m: float,
    use_klt: bool,
    use_flow: bool,
    dropout_indices: Optional[Set[int]] = None
) -> pd.DataFrame:
    """
    Process frames to detect tags using hybrid strategy.

    Args:
        dropout_indices: Set of frame indices (0-based relative to df_frames)
                         where AprilTag detection should be artificially failed.
    """

    # State variables
    prev_img = None
    prev_corners = None
    last_known_tag_id = -1

    rows = []

    if dropout_indices is None:
        dropout_indices = set()

    # If run via direct execution, tqdm might be useful.
    # But if called from another script, maybe silent?
    # We will use tqdm if dataframe is large enough or maybe pass verbose flag.
    # For now, just use tqdm.

    iterator = tqdm(df_frames.iterrows(), total=len(df_frames), desc=f"Robust Detect {cam_id}")

    for idx, (_, r) in enumerate(iterator):
        t_sec = float(r["t_sec"])
        img_path = Path(r["image_path"])

        # Read image
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            # Add failed row
            rows.append({T: t_sec, CAM: cam_id, DETECTED: 0, SOURCE: "none"})
            prev_img = None
            prev_corners = None
            continue

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. Try Standard Detection (unless simulated dropout)
        best = None
        if idx not in dropout_indices:
            dets = detect_tags(img_gray, family=family, params=det_params)
            best = pick_best_single_tag(dets, tag_id=tag_id)

        current_corners = None
        source = "none"
        decision_margin = np.nan
        hamming = np.nan
        tag_area = np.nan
        klt_err = np.nan
        current_tag_id = -1

        if best is not None:
            # SUCCESS
            current_corners = best.corners_px
            source = "tag"
            decision_margin = best.decision_margin
            hamming = best.hamming
            tag_area = best.tag_area_px2
            current_tag_id = best.tag_id
            last_known_tag_id = current_tag_id
        elif use_klt and prev_corners is not None and prev_img is not None:
            # FAILURE -> Attempt KLT Fallback
            new_corners, success, err = track_corners_lk(prev_img, img_gray, prev_corners)
            if success:
                current_corners = new_corners
                source = "klt"
                klt_err = err
                # Estimate area for continuity check or metric
                tag_area = cv2.contourArea(current_corners.astype(np.float32))
                # Persist ID
                current_tag_id = last_known_tag_id
            else:
                source = "lost"
        else:
            source = "none"

        # 2. Solve PnP
        pose_data = {
             X: np.nan, Y: np.nan, Z: np.nan,
             QX: np.nan, QY: np.nan, QZ: np.nan, QW: np.nan,
             REPROJ_ERR: np.nan
        }

        if source in ["tag", "klt"] and current_corners is not None:
            pose, reproj = solve_pnp_from_corners(current_corners, caminfo, tag_size_m, use_ransac=False)
            pose_data[X] = pose.x_m
            pose_data[Y] = pose.y_m
            pose_data[Z] = pose.z_m
            pose_data[QX] = pose.qx
            pose_data[QY] = pose.qy
            pose_data[QZ] = pose.qz
            pose_data[QW] = pose.qw
            pose_data[REPROJ_ERR] = reproj

        # 3. Compute Flow (MCI proxy)
        flow_mag = np.nan
        if use_flow and prev_img is not None:
            # We need an ROI. Use current corners if available, else prev corners?
            # Ideally use current ROI estimate (even if from KLT).
            # If completely lost, use prev corners as best guess for where it WAS?
            # Or skip flow if lost.
            roi_to_use = current_corners if current_corners is not None else prev_corners

            if roi_to_use is not None:
                flow_mag = compute_flow_roi(prev_img, img_gray, roi_to_use)

        # Update State
        if source in ["tag", "klt"]:
            prev_corners = current_corners
        else:
            prev_corners = None # Reset if lost to avoid drifting forever from stale data?
            pass

        prev_img = img_gray

        # Record
        row = {
            T: t_sec,
            CAM: cam_id,
            TAG: int(tag_id) if tag_id else int(current_tag_id),
            **pose_data,
            DECISION_MARGIN: decision_margin,
            HAMMING: hamming,
            TAG_AREA: tag_area,
            DETECTED: 1 if source in ["tag", "klt"] else 0,
            SOURCE: source,
            KLT_RMSE: klt_err,
            FLOW_MAG: flow_mag
        }
        rows.append(row)

    df_out = pd.DataFrame(rows)
    df_out = ensure_columns(df_out, [T, CAM, TAG, X, Y, Z, QX, QY, QZ, QW, DETECTED, SOURCE, FLOW_MAG, KLT_RMSE])
    return df_out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="results/<run> directory")
    ap.add_argument("--cam", required=True, help="e.g., sony_cam1")
    ap.add_argument("--family", default="tag36h11")
    ap.add_argument("--tag-size-m", type=float, default=0.02)
    ap.add_argument("--tag-id", type=int, default=None)
    ap.add_argument("--out", default=None, help="Output CSV path.")
    ap.add_argument("--legacy-out", default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--nthreads", type=int, default=4)
    # Robustness params
    ap.add_argument("--use-klt", action="store_true", default=True, help="Enable KLT fallback")
    ap.add_argument("--use-flow", action="store_true", default=True, help="Compute optical flow mag")

    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    export_dir = run_dir / "export" / args.cam
    frames_csv = export_dir / "frames.csv"
    caminfo_yaml = export_dir / "camera_info.yaml"

    if not frames_csv.exists():
        raise FileNotFoundError(f"Missing frames.csv: {frames_csv}")
    caminfo = load_camera_info_yaml(caminfo_yaml)

    out_csv = Path(args.out) if args.out else (run_dir / "detect" / "robust" / f"{args.cam}_poses.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    det_params = {
        "nthreads": args.nthreads,
        "quad_decimate": 1.0,
        "quad_sigma": 0.8,
        "refine_edges": True,
        "decode_sharpening": 0.25,
        "debug": False,
    }

    df_frames = pd.read_csv(frames_csv)
    if args.max_frames is not None:
        df_frames = df_frames.iloc[: args.max_frames].copy()

    df_out = process_frames(
        df_frames=df_frames,
        caminfo=caminfo,
        cam_id=args.cam,
        tag_id=args.tag_id,
        det_params=det_params,
        family=args.family,
        tag_size_m=args.tag_size_m,
        use_klt=args.use_klt,
        use_flow=args.use_flow
    )

    df_out.to_csv(out_csv, index=False)
    print(f"✅ Wrote robust detection CSV: {out_csv}")

    if args.legacy_out:
        export_legacy_detect_csv(df_out, args.legacy_out)

if __name__ == "__main__":
    main()
