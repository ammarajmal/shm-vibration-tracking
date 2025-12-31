#!/usr/bin/env python3
"""
Sync cameras script.

Loads world-transformed CSVs, calculates offsets, and resamples to a master time grid.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import yaml
from rich.console import Console
from rich.table import Table

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from shmtrack.io.schema import (
    T, CAM, POSE_WORLD_COLUMNS, Y,
    ensure_columns
)
from shmtrack.sync.resample import create_master_time_grid, resample_dataframe
from shmtrack.sync.drift import estimate_offset_cross_corr, align_timestamps

console = Console()

def load_world_csvs(run_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load world_{cam}.csv files from results/<run>/world/
    """
    world_dir = run_dir / "world"
    if not world_dir.exists():
        console.print(f"[red]World directory not found: {world_dir}[/red]")
        return {}

    dfs = {}
    for f in world_dir.glob("world_*.csv"):
        # Expect filename: world_sony_camX.csv
        cam_name = f.stem.replace("world_", "")
        console.print(f"Loading {f.name} as {cam_name}...")
        df = pd.read_csv(f)
        df = ensure_columns(df, POSE_WORLD_COLUMNS, fill_value=np.nan)
        dfs[cam_name] = df

    return dfs

def save_synced_csvs(dfs: Dict[str, pd.DataFrame], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for cam, df in dfs.items():
        out_path = out_dir / f"synced_{cam}.csv"
        df.to_csv(out_path, index=False)
        console.print(f"Saved {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Synchronize and resample multi-camera data.")
    parser.add_argument("run_dir", type=Path, help="Path to results directory for a run (e.g. results/experiment_1)")
    parser.add_argument("--fps", type=float, default=60.0, help="Target FPS for resampling")
    parser.add_argument("--align", action="store_true", help="Perform cross-correlation alignment")
    parser.add_argument("--master", type=str, default="sony_cam1", help="Master camera for alignment")
    parser.add_argument("--max-lag", type=float, default=2.0, help="Max lag in seconds for alignment search")
    parser.add_argument("--gap-threshold", type=float, default=0.2, help="Gap threshold in seconds to mask output")

    args = parser.parse_args()

    console.rule("[bold blue]Stage C.4: Multi-Camera Synchronization")

    # 1. Load Data
    dfs = load_world_csvs(args.run_dir)
    if not dfs:
        console.print("[red]No data loaded. Exiting.[/red]")
        return

    # 2. Alignment (Optional)
    offsets = {cam: 0.0 for cam in dfs.keys()}

    if args.align:
        if args.master not in dfs:
            console.print(f"[red]Master camera {args.master} not found in loaded data.[/red]")
            return

        master_df = dfs[args.master]
        console.print(f"Aligning to master: {args.master}")

        table = Table(title="Estimated Offsets")
        table.add_column("Camera")
        table.add_column("Offset (s)")

        for cam, df in dfs.items():
            if cam == args.master:
                table.add_row(cam, "0.0000")
                continue

            offset = estimate_offset_cross_corr(
                master_df, df,
                col=Y, # Vertical displacement usually best for vibration
                max_lag_sec=args.max_lag
            )
            offsets[cam] = offset
            table.add_row(cam, f"{offset:+.4f}")

        console.print(table)

        # Apply offsets
        dfs = align_timestamps(dfs, offsets)

    # 3. Create Master Grid
    t_grid = create_master_time_grid(dfs, fps=args.fps)
    console.print(f"Master time grid: {len(t_grid)} samples, {t_grid[0]:.2f}s to {t_grid[-1]:.2f}s ({args.fps} Hz)")

    # 4. Resample
    synced_dfs = {}
    for cam, df in dfs.items():
        console.print(f"Resampling {cam}...")
        synced_dfs[cam] = resample_dataframe(
            df, t_grid,
            kind='linear', # Linear is robust
            gap_threshold_sec=args.gap_threshold
        )

    # 5. Save
    save_dir = args.run_dir / "sync"
    save_synced_csvs(synced_dfs, save_dir)

    # Save metadata
    meta = {
        "fps": args.fps,
        "master_cam": args.master,
        "offsets": offsets,
        "aligned": args.align
    }
    with open(save_dir / "sync_metadata.yaml", "w") as f:
        yaml.dump(meta, f)

    console.print("[green]Synchronization complete.[/green]")

if __name__ == "__main__":
    main()
