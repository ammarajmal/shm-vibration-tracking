#!/usr/bin/env python3
"""
Simulate dropout (stress test) for robust tracking.

1. Loads a run (frames).
2. Runs robust detection with ARTIFICIAL gaps.
3. Compares result to Ground Truth (clean run, or robust run without gaps).
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from rich.console import Console
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from shmtrack.io.schema import (
    T, Y, SOURCE, KLT_RMSE, DETECTED
)
# Import logic from detect_robust (as a module)
# Assuming detect_robust.py is in same dir as this script
from detect_robust import process_frames, load_camera_info_yaml

console = Console()

def generate_dropout_mask(n_frames: int, gap_len: int, period: int) -> set[int]:
    """
    Generate set of indices to drop.
    Drop 'gap_len' frames every 'period' frames.
    """
    mask = set()
    for start in range(period, n_frames, period):
        end = min(start + gap_len, n_frames)
        for i in range(start, end):
            mask.add(i)
    return mask

def main():
    parser = argparse.ArgumentParser(description="Simulate dropout and evaluate recovery.")
    parser.add_argument("run_dir", type=Path, help="Run directory")
    parser.add_argument("--cam", required=True)
    parser.add_argument("--gap-len", type=int, default=5, help="Length of gap in frames")
    parser.add_argument("--period", type=int, default=60, help="Gap period in frames")
    parser.add_argument("--baseline-csv", type=Path, default=None, help="Ground truth CSV (optional)")

    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    export_dir = run_dir / "export" / args.cam
    frames_csv = export_dir / "frames.csv"
    caminfo_yaml = export_dir / "camera_info.yaml"

    if not frames_csv.exists():
        console.print(f"[red]Missing {frames_csv}[/red]")
        return

    df_frames = pd.read_csv(frames_csv)
    caminfo = load_camera_info_yaml(caminfo_yaml)

    det_params = {
        "nthreads": 4,
        "quad_decimate": 1.0,
        "quad_sigma": 0.8,
        "refine_edges": True,
        "decode_sharpening": 0.25,
        "debug": False,
    }

    # 1. Run Ground Truth (Full Detection) if not provided
    # Or assuming the user wants to compare "Robust with gaps" vs "Robust without gaps"
    # Ideally GT is standard detection on clean data.

    df_gt = None
    if args.baseline_csv:
        df_gt = pd.read_csv(args.baseline_csv)
    else:
        console.print("Running Ground Truth Detection (Clean)...")
        df_gt = process_frames(
            df_frames, caminfo, args.cam, None, det_params,
            family="tag36h11", tag_size_m=0.02, use_klt=True, use_flow=False, dropout_indices=None
        )

    # 2. Generate Mask
    mask = generate_dropout_mask(len(df_frames), args.gap_len, args.period)
    console.print(f"Generated dropout mask: {len(mask)} frames dropped out of {len(df_frames)}.")

    # 3. Run Robust Detection with Gaps
    console.print("Running Robust Detection with Simulated Gaps...")
    df_robust = process_frames(
        df_frames, caminfo, args.cam, None, det_params,
        family="tag36h11", tag_size_m=0.02, use_klt=True, use_flow=False, dropout_indices=mask
    )

    # 4. Analysis
    # Align rows (assume T matches if frames match)
    # Filter only rows that were dropped

    # Create comparison DF
    # We rely on indices being consistent as process_frames preserves frame order

    # Check simple RMSE on Y
    y_gt = df_gt[Y].fillna(0).values
    y_rob = df_robust[Y].fillna(0).values

    # Calculate RMSE only on gap indices
    gap_indices = sorted(list(mask))
    if not gap_indices:
        console.print("No gaps generated?")
        return

    # Filter indices where GT was actually detected
    valid_gt_mask = df_gt[DETECTED].values.astype(bool)

    eval_indices = [i for i in gap_indices if i < len(y_gt) and valid_gt_mask[i]]

    if not eval_indices:
        console.print("No valid GT data in gap regions to evaluate against.")
        return

    rmse = np.sqrt(np.mean((y_gt[eval_indices] - y_rob[eval_indices])**2))
    max_err = np.max(np.abs(y_gt[eval_indices] - y_rob[eval_indices]))

    console.print(f"[bold green]Results (on {len(eval_indices)} gap frames):[/bold green]")
    console.print(f"RMSE (Y): {rmse:.6f} m")
    console.print(f"Max Error (Y): {max_err:.6f} m")

    recovered_count = df_robust.iloc[eval_indices][DETECTED].sum()
    recovery_rate = recovered_count / len(eval_indices)
    console.print(f"Recovery Rate: {recovery_rate*100:.1f}%")

    # 5. Plot
    out_dir = run_dir / "sim_results"
    out_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(df_gt[T], df_gt[Y], 'k-', alpha=0.3, label="Ground Truth", linewidth=2)

    # Highlight gaps
    gap_t = df_gt.iloc[gap_indices][T].values
    gap_y = df_gt.iloc[gap_indices][Y].values
    plt.plot(gap_t, gap_y, 'rx', label="Dropped Frames (GT)")

    plt.plot(df_robust[T], df_robust[Y], 'b--', label="Robust Recovery (KLT)", linewidth=1)

    plt.title(f"Dropout Stress Test (Gap {args.gap_len}, Period {args.period})")
    plt.xlabel("Time (s)")
    plt.ylabel("Y (m)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_dir / f"dropout_gap{args.gap_len}_cam{args.cam}.png")
    console.print(f"Plot saved to {out_dir}")

    # Save CSVs
    df_robust.to_csv(out_dir / f"robust_gap{args.gap_len}.csv", index=False)

if __name__ == "__main__":
    main()
