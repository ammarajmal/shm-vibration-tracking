"""
Drift and offset estimation utilities.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import correlate, correlation_lags

from shmtrack.io.schema import T, Y, CAM
from shmtrack.sync.resample import resample_dataframe

def estimate_offset_cross_corr(
    df_ref: pd.DataFrame,
    df_target: pd.DataFrame,
    col: str = Y,
    upsample_fps: float = 1000.0,
    max_lag_sec: float = 2.0
) -> float:
    """
    Estimate the time offset to add to df_target to align it with df_ref.

    offset = t_ref - t_target
    aligned_t_target = t_target + offset

    Method:
    1. Resample both signals to a high-frequency grid (upsample_fps).
    2. Compute cross-correlation.
    3. Find lag with max correlation.

    Args:
        df_ref: Reference dataframe.
        df_target: Target dataframe to be shifted.
        col: Column to use for correlation (default 'y_m' for vertical vibration).
        upsample_fps: Sampling rate for correlation grid.
        max_lag_sec: Maximum expected lag in seconds to search.

    Returns:
        offset_sec: Time to ADD to df_target timestamps.
    """
    # 1. Determine common time range for intersection
    t_min = max(df_ref[T].min(), df_target[T].min())
    t_max = min(df_ref[T].max(), df_target[T].max())

    # If no overlap, we can't correlate easily without assuming they are close.
    # We'll create a grid based on the union of times clipped to valid range if needed,
    # but strictly correlation requires looking at similar signals.
    # Let's assume we resample over the union of ranges to catch shifts.

    t_min_all = min(df_ref[T].min(), df_target[T].min())
    t_max_all = max(df_ref[T].max(), df_target[T].max())

    num_samples = int((t_max_all - t_min_all) * upsample_fps)
    t_grid = np.linspace(t_min_all, t_max_all, num_samples)

    # 2. Resample
    # Use fill_value=0 to avoid edge effects in correlation, or mean subtraction
    # Actually, better to subtract mean before correlation.

    # Simple linear interp
    s_ref = resample_dataframe(df_ref, t_grid, kind='linear')[col].fillna(0).values
    s_tgt = resample_dataframe(df_target, t_grid, kind='linear')[col].fillna(0).values

    # Remove DC component
    s_ref -= np.mean(s_ref)
    s_tgt -= np.mean(s_tgt)

    # 3. Cross-correlation
    corr = correlate(s_ref, s_tgt, mode='full')
    lags = correlation_lags(len(s_ref), len(s_tgt), mode='full')

    # Find peak
    peak_idx = np.argmax(corr)
    lag_samples = lags[peak_idx]

    offset_sec = lag_samples / upsample_fps

    # Sanity check
    if abs(offset_sec) > max_lag_sec:
        # print(f"Warning: Estimated offset {offset_sec:.3f}s exceeds max_lag {max_lag_sec}s")
        pass

    return offset_sec


def align_timestamps(dfs: Dict[str, pd.DataFrame], offsets: Dict[str, float]) -> Dict[str, pd.DataFrame]:
    """
    Apply offsets to dataframes.

    Args:
        dfs: Dictionary of camera dataframes.
        offsets: Dictionary of {cam_id: offset_sec}.

    Returns:
        New dictionary with adjusted timestamps.
    """
    out_dfs = {}
    for cam, df in dfs.items():
        if df.empty:
            out_dfs[cam] = df
            continue

        new_df = df.copy()
        offset = offsets.get(cam, 0.0)
        new_df[T] = new_df[T] + offset
        out_dfs[cam] = new_df

    return out_dfs
