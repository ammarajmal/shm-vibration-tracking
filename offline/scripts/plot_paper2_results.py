#!/usr/bin/env python3
"""
Visualize Paper 2 Results: Robustness and Fusion.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from shmtrack.io.schema import T, Y, MCI, FLOW_MAG

def plot_recovery(df_final: pd.DataFrame, df_cam: pd.DataFrame, out_path: Path):
    """
    Plot recovery from dropout.
    Show Raw Camera Y (with gaps) vs Fused Y (smooth).
    """
    plt.figure(figsize=(10, 5))

    # Raw Cam (filter gaps if possible or show gaps)
    # df_cam might have NaNs or explicit gaps.
    t_cam = df_cam[T]
    y_cam = df_cam[Y]

    plt.plot(t_cam, y_cam, 'o', markersize=2, label="Raw Camera Input", alpha=0.5, color='gray')

    # Fused
    t_fused = df_final[T]
    y_fused = df_final[Y]

    plt.plot(t_fused, y_fused, '-', linewidth=1.5, label="Robust Fused Output", color='blue')

    plt.xlabel("Time (s)")
    plt.ylabel("Displacement Y (m)")
    plt.title("Robustness to Dropouts (Paper 2)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_validation(df_cam: pd.DataFrame, out_path: Path):
    """
    Plot v_pose vs v_flow and MCI.
    """
    if 'v_pose' not in df_cam.columns or 'v_flow_scaled' not in df_cam.columns:
        print("Skipping validation plot (missing columns)")
        return

    t = df_cam[T]
    v_pose = df_cam['v_pose']
    v_flow = df_cam['v_flow_scaled']
    mci = df_cam[MCI]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(t, v_pose, 'b-', label='Pose Velocity')
    ax1.plot(t, v_flow, 'g--', label='Flow Velocity (Scaled)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Velocity', color='k')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(t, mci, 'r-', alpha=0.3, label='MCI')
    ax2.set_ylabel('MCI Score', color='r')
    ax2.set_ylim(0, 5) # Clip high MCI
    ax2.legend(loc='upper right')

    plt.title("Motion Consistency Index (MCI) Validation")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    # Load data
    fused_csv = args.run_dir / "fused_final.csv"
    if not fused_csv.exists():
        print("Fused data not found.")
        return

    df_fused = pd.read_csv(fused_csv)

    # Load one analyzed camera for detail view
    sync_dir = args.run_dir / "sync"
    analyzed_files = list(sync_dir.glob("analyzed_*.csv"))
    if not analyzed_files:
        print("No analyzed camera files found.")
        return

    df_cam = pd.read_csv(analyzed_files[0])

    # Plots
    plot_dir = args.run_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    plot_recovery(df_fused, df_cam, plot_dir / "paper2_recovery.png")
    plot_validation(df_cam, plot_dir / "paper2_mci_validation.png")

    print(f"Plots saved to {plot_dir}")

if __name__ == "__main__":
    main()
