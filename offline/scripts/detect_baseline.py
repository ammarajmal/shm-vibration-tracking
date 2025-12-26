#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import cv2
from tqdm import tqdm

from shmtrack.io.schema import (
    T, CAM, TAG, X, Y, Z, QX, QY, QZ, QW,
    DECISION_MARGIN, HAMMING, TAG_AREA,
    DETECTED, SOURCE, REPROJ_ERR,
    export_legacy_detect_csv,
)
from shmtrack.vision.apriltag_pupil import detect_tags, pick_best_single_tag
from shmtrack.vision.pnp_pose import solve_pnp_from_corners
from shmtrack.io.rosbag_reader import CameraInfoData


def load_camera_info_yaml(caminfo_path: Path) -> CameraInfoData:
    import yaml
    d = yaml.safe_load(caminfo_path.read_text())
    import numpy as np
    return CameraInfoData(
        K=np.array(d["K"], dtype=float),
        D=np.array(d["D"], dtype=float),
        width=int(d.get("width", 0)),
        height=int(d.get("height", 0)),
        frame_id=str(d.get("frame_id", "")),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="results/<run> directory containing export/<cam>/frames.csv")
    ap.add_argument("--cam", required=True, help="e.g., sony_cam1")
    ap.add_argument("--family", default="tag36h11")
    ap.add_argument("--tag-size-m", type=float, default=0.02)
    ap.add_argument("--tag-id", type=int, default=None, help="If set, only accept this tag id. Else pick best.")
    ap.add_argument("--out", default=None, help="Output CSV path. Default: run-dir/detect/baseline/<cam>_poses.csv")
    ap.add_argument("--legacy-out", default=None, help="Optional legacy-format CSV output path")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--nthreads", type=int, default=4)
    ap.add_argument("--quad-decimate", type=float, default=1.0)
    ap.add_argument("--quad-sigma", type=float, default=0.8)
    ap.add_argument("--refine-edges", action="store_true", default=True)
    ap.add_argument("--decode-sharpening", type=float, default=0.25)
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    export_dir = run_dir / "export" / args.cam
    frames_csv = export_dir / "frames.csv"
    caminfo_yaml = export_dir / "camera_info.yaml"

    if not frames_csv.exists():
        raise FileNotFoundError(f"Missing frames.csv: {frames_csv}")
    if not caminfo_yaml.exists():
        raise FileNotFoundError(f"Missing camera_info.yaml: {caminfo_yaml}")

    caminfo = load_camera_info_yaml(caminfo_yaml)

    out_csv = Path(args.out) if args.out else (run_dir / "detect" / "baseline" / f"{args.cam}_poses.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    det_params = {
        "nthreads": args.nthreads,
        "quad_decimate": args.quad_decimate,
        "quad_sigma": args.quad_sigma,
        "refine_edges": True,  # keep stable default
        "decode_sharpening": args.decode_sharpening,
        "debug": False,
    }

    df_frames = pd.read_csv(frames_csv)
    if args.max_frames is not None:
        df_frames = df_frames.iloc[: args.max_frames].copy()

    rows = []
    for _, r in tqdm(df_frames.iterrows(), total=len(df_frames), desc=f"Detect {args.cam}"):
        t_sec = float(r["t_sec"])
        img_path = Path(r["image_path"])
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue

        dets = detect_tags(img, family=args.family, params=det_params)
        best = pick_best_single_tag(dets, tag_id=args.tag_id)

        if best is None:
            rows.append({
                T: t_sec, CAM: args.cam, TAG: args.tag_id if args.tag_id is not None else -1,
                X: "", Y: "", Z: "",
                QX: "", QY: "", QZ: "", QW: "",
                DECISION_MARGIN: "", HAMMING: "", TAG_AREA: "",
                DETECTED: 0, SOURCE: "none", REPROJ_ERR: "",
            })
            continue

        pose, reproj = solve_pnp_from_corners(best.corners_px, caminfo, args.tag_size_m, use_ransac=False)

        rows.append({
            T: t_sec, CAM: args.cam, TAG: int(best.tag_id),
            X: pose.x_m, Y: pose.y_m, Z: pose.z_m,
            QX: pose.qx, QY: pose.qy, QZ: pose.qz, QW: pose.qw,
            DECISION_MARGIN: best.decision_margin,
            HAMMING: best.hamming,
            TAG_AREA: best.tag_area_px2,
            DETECTED: 1, SOURCE: "tag", REPROJ_ERR: reproj,
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_csv, index=False)
    print(f"✅ Wrote canonical detection CSV: {out_csv}")

    if args.legacy_out:
        export_legacy_detect_csv(df_out, args.legacy_out)
        print(f"✅ Wrote legacy detection CSV: {args.legacy_out}")


if __name__ == "__main__":
    main()
