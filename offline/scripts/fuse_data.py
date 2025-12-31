#!/usr/bin/env python3
"""
Fusion script.

Loads synchronized camera data, calculates MCI, weights, and fuses them.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from rich.console import Console

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from shmtrack.fusion.mci import compute_mci
from shmtrack.fusion.weighted import compute_weights, fuse_multicam_weighted_avg
from shmtrack.io.schema import ensure_columns, T, X, Y, Z, QX, QY, QZ, QW

console = Console()

def main():
    parser = argparse.ArgumentParser(description="Fuse synchronized multi-camera data.")
    parser.add_argument("run_dir", type=Path, help="Run directory (results/experiment_X)")
    parser.add_argument("--smooth-win", type=int, default=11, help="Savitzky-Golay window size")
    parser.add_argument("--smooth-poly", type=int, default=3, help="Savitzky-Golay poly order")
    parser.add_argument("--weight-tag", type=float, default=1.0)
    parser.add_argument("--weight-klt", type=float, default=0.5)

    args = parser.parse_args()

    sync_dir = args.run_dir / "sync"
    if not sync_dir.exists():
        console.print(f"[red]Sync directory not found: {sync_dir}[/red]")
        return

    # Load synchronized CSVs
    dfs = []
    cam_files = sorted(list(sync_dir.glob("synced_*.csv")))

    if not cam_files:
        console.print("[red]No synced CSVs found.[/red]")
        return

    console.print(f"Found {len(cam_files)} synced camera files.")

    processed_dfs = []

    for f in cam_files:
        console.print(f"Processing {f.name}...")
        df = pd.read_csv(f)

        # 1. Compute MCI
        df = compute_mci(df)

        # 2. Compute Weights
        df = compute_weights(
            df,
            base_weight_tag=args.weight_tag,
            base_weight_klt=args.weight_klt
        )

        # Save intermediate per-camera analysis
        out_name = f.name.replace("synced_", "analyzed_")
        df.to_csv(sync_dir / out_name, index=False)

        processed_dfs.append(df)

    # 3. Fuse
    console.print("Fusing data...")
    fused_df = fuse_multicam_weighted_avg(
        processed_dfs,
        smooth_window=args.smooth_win,
        smooth_poly=args.smooth_poly
    )

    # Save Final Fused
    out_path = args.run_dir / "fused_final.csv"
    fused_df.to_csv(out_path, index=False)
    console.print(f"[green]Saved fused data to {out_path}[/green]")

if __name__ == "__main__":
    main()
