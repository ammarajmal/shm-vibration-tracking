"""
Resampling and synchronization utilities.
"""
from __future__ import annotations

from typing import Dict, Optional, List

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

from shmtrack.io.schema import (
    T, X, Y, Z,
    QX, QY, QZ, QW,
    DETECTED, SOURCE,
    DECISION_MARGIN, TAG_AREA, REPROJ_ERR,
    CAM, TAG,
    ensure_columns
)


def create_master_time_grid(dfs: Dict[str, pd.DataFrame], fps: float = 60.0) -> np.ndarray:
    """
    Create a common time grid covering the span of all provided dataframes.
    """
    valid_dfs = [df for df in dfs.values() if not df.empty and T in df.columns]
    if not valid_dfs:
        raise ValueError("No valid dataframes provided to create time grid.")

    t_min = min(df[T].min() for df in valid_dfs)
    t_max = max(df[T].max() for df in valid_dfs)

    # Calculate number of samples to hit the target FPS exactly
    duration = t_max - t_min
    num_samples = int(np.round(duration * fps)) + 1

    return np.linspace(t_min, t_max, num_samples)


def resample_dataframe(
    df: pd.DataFrame,
    target_time: np.ndarray,
    kind: str = 'linear',
    gap_threshold_sec: Optional[float] = None
) -> pd.DataFrame:
    """
    Resample a single camera dataframe onto the target time grid.

    - Translations (x, y, z): Interpolated (linear/cubic)
    - Rotations (qx, qy, qz, qw): Slerp
    - Scalars (decision_margin, etc.): Interpolated
    - Categorical/Flags (cam, tag, source): Nearest/Forward fill or preserved from context

    Args:
        df: Input dataframe (must have T column).
        target_time: Target timestamps.
        kind: Interpolation kind ('linear', 'cubic', etc.) for translation.
        gap_threshold_sec: If provided, gaps in source data larger than this
                           will result in NaN in the output for that segment.
    """
    if df.empty:
        return pd.DataFrame(columns=df.columns)

    # Sort just in case
    df = df.sort_values(T)

    src_t = df[T].values

    # Initialize output dataframe
    out_df = pd.DataFrame({T: target_time})

    # 1. Interpolate Translation
    translation_cols = [c for c in [X, Y, Z] if c in df.columns]
    for col in translation_cols:
        f = interp1d(src_t, df[col].values, kind=kind, bounds_error=False, fill_value=np.nan)
        out_df[col] = f(target_time)

    # 2. Interpolate Rotation (Slerp)
    if all(c in df.columns for c in [QX, QY, QZ, QW]):
        quats = df[[QX, QY, QZ, QW]].values
        try:
            rotations = Rotation.from_quat(quats)
            slerp = Slerp(src_t, rotations)
            interp_rots = slerp(target_time)
            interp_quats = interp_rots.as_quat()

            out_df[QX] = interp_quats[:, 0]
            out_df[QY] = interp_quats[:, 1]
            out_df[QZ] = interp_quats[:, 2]
            out_df[QW] = interp_quats[:, 3]
        except Exception as e:
            out_df[QX] = np.nan
            out_df[QY] = np.nan
            out_df[QZ] = np.nan
            out_df[QW] = np.nan

    # 3. Interpolate Auxiliary Scalars
    aux_cols = [c for c in [DECISION_MARGIN, TAG_AREA, REPROJ_ERR] if c in df.columns]
    for col in aux_cols:
        f = interp1d(src_t, df[col].values, kind='linear', bounds_error=False, fill_value=np.nan)
        out_df[col] = f(target_time)

    # 4. Handle Categorical / Metadata
    if CAM in df.columns:
        out_df[CAM] = df[CAM].iloc[0]
    if TAG in df.columns:
        out_df[TAG] = df[TAG].iloc[0]

    # 'detected': if any translation column is present, use it to determine detection status (NaN check)
    # If no translation columns, use just T range check?
    # For now, if we have X, use X. If not, use Y, etc.
    check_col = None
    if X in out_df.columns: check_col = X
    elif Y in out_df.columns: check_col = Y
    elif Z in out_df.columns: check_col = Z

    if check_col:
        out_df[DETECTED] = out_df[check_col].notna().astype(int)
    else:
        # If no data columns to interpolate, we can't really say if detected.
        # But maybe we just check if time is within bounds of original signal?
        # Let's assume DETECTED=0 if we didn't output any data.
        out_df[DETECTED] = 0

    out_df[SOURCE] = "interp"

    # 5. Gap Handling (Masking)
    if gap_threshold_sec is not None:
        dt = np.diff(src_t)
        gap_indices = np.where(dt > gap_threshold_sec)[0]

        for idx in gap_indices:
            t_gap_start = src_t[idx]
            t_gap_end = src_t[idx+1]

            mask = (out_df[T] > t_gap_start) & (out_df[T] < t_gap_end)

            # Mask pose columns
            cols_to_mask = [c for c in [X, Y, Z, QX, QY, QZ, QW] if c in out_df.columns]
            out_df.loc[mask, cols_to_mask] = np.nan
            out_df.loc[mask, DETECTED] = 0

    return out_df
