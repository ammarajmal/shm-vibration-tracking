"""
Weighted fusion logic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from shmtrack.io.schema import (
    SOURCE, MCI, DETECTED,
    W_TAG, W_KLT,
    X, Y, Z, QX, QY, QZ, QW
)

def compute_weights(
    df: pd.DataFrame,
    base_weight_tag: float = 1.0,
    base_weight_klt: float = 0.5
) -> pd.DataFrame:
    """
    Compute fusion weights based on source type and MCI score.

    Args:
        df: Input dataframe (must have SOURCE and MCI columns).
        base_weight_tag: Base confidence for AprilTag detections.
        base_weight_klt: Base confidence for KLT tracked points.

    Returns:
        DataFrame with 'weight' column appended.
    """
    df = df.copy()

    # Initialize weights
    w = np.zeros(len(df))

    # Assign base weights
    if SOURCE in df.columns:
        w[df[SOURCE] == 'tag'] = base_weight_tag
        w[df[SOURCE] == 'klt'] = base_weight_klt
        w[df[SOURCE] == 'fused'] = base_weight_tag # Treat existing fused as trusted?
    else:
        # Default to tag weight if source unknown but detected
        if DETECTED in df.columns:
            w[df[DETECTED] == 1] = base_weight_tag

    # Apply MCI Penalty
    # Weight = Base * (1 - MCI)
    # Clip MCI to [0, 1] just in case
    if MCI in df.columns:
        mci = df[MCI].fillna(1.0).clip(0, 1) # If NaN, assume bad (1.0) -> weight 0
        w = w * (1.0 - mci)

    df['weight'] = w

    # Store parameters used?
    df[W_TAG] = base_weight_tag
    df[W_KLT] = base_weight_klt

    return df

def fuse_multicam_weighted_avg(
    dfs: list[pd.DataFrame],
    smooth_window: int = 11,
    smooth_poly: int = 3
) -> pd.DataFrame:
    """
    Fuse multiple synchronized camera dataframes into one world pose using weighted average.

    Args:
        dfs: List of camera dataframes (must have 'weight' and world pose columns).
             Assumes they are already on the same time grid (synced).
        smooth_window: Window length for Savitzky-Golay filter.
        smooth_poly: Poly order for Savitzky-Golay filter.

    Returns:
        Fused dataframe.
    """
    if not dfs:
        return pd.DataFrame()

    # Use the first df's time as reference
    t_ref = dfs[0]['t_sec'].values
    n_samples = len(t_ref)

    # Accumulators
    sum_w_x = np.zeros(n_samples)
    sum_w_y = np.zeros(n_samples)
    sum_w_z = np.zeros(n_samples)
    sum_w = np.zeros(n_samples)

    # Rotation averaging is complex.
    # For small vibrations, linear averaging of quaternions (renormalized) is roughly acceptable,
    # or averaging Euler angles if aligned.
    # Or just taking the best camera?
    # Let's do weighted average of position. For rotation, maybe just take the one with highest weight?
    # Or weighted avg of quaternions (chordal L2 mean approximation).

    sum_w_qx = np.zeros(n_samples)
    sum_w_qy = np.zeros(n_samples)
    sum_w_qz = np.zeros(n_samples)
    sum_w_qw = np.zeros(n_samples)

    for df in dfs:
        # Ensure aligned length (should be if synced)
        # If not, we might need to reindex/align.
        # Assuming strictly aligned from sync_cameras.py

        # Get weight
        w = df['weight'].fillna(0.0).values

        # Position
        x = df[X].fillna(0.0).values
        y = df[Y].fillna(0.0).values
        z = df[Z].fillna(0.0).values

        sum_w_x += w * x
        sum_w_y += w * y
        sum_w_z += w * z
        sum_w += w

        # Rotation
        qx = df[QX].fillna(0.0).values
        qy = df[QY].fillna(0.0).values
        qz = df[QZ].fillna(0.0).values
        qw = df[QW].fillna(0.0).values # Identity is 0,0,0,1? Or whatever.

        # Ensure canonical hemisphere for quaternions to avoid averaging q and -q
        # Dot product with reference (e.g. first valid q) > 0
        # For simplicity, just add weighted components.
        sum_w_qx += w * qx
        sum_w_qy += w * qy
        sum_w_qz += w * qz
        sum_w_qw += w * qw

    # Normalize
    # Handle zero weight
    mask_valid = sum_w > 1e-6

    fused_x = np.zeros(n_samples)
    fused_y = np.zeros(n_samples)
    fused_z = np.zeros(n_samples)

    fused_x[mask_valid] = sum_w_x[mask_valid] / sum_w[mask_valid]
    fused_y[mask_valid] = sum_w_y[mask_valid] / sum_w[mask_valid]
    fused_z[mask_valid] = sum_w_z[mask_valid] / sum_w[mask_valid]

    # Rotation normalization
    fused_qx = sum_w_qx
    fused_qy = sum_w_qy
    fused_qz = sum_w_qz
    fused_qw = sum_w_qw

    norm_q = np.sqrt(fused_qx**2 + fused_qy**2 + fused_qz**2 + fused_qw**2)
    mask_q = norm_q > 1e-6

    fused_qx[mask_q] /= norm_q[mask_q]
    fused_qy[mask_q] /= norm_q[mask_q]
    fused_qz[mask_q] /= norm_q[mask_q]
    fused_qw[mask_q] /= norm_q[mask_q]
    # Default to identity where undefined?
    fused_qw[~mask_q] = 1.0

    # Create result DF
    res_df = pd.DataFrame({
        't_sec': t_ref,
        X: fused_x,
        Y: fused_y,
        Z: fused_z,
        QX: fused_qx,
        QY: fused_qy,
        QZ: fused_qz,
        QW: fused_qw,
        'total_weight': sum_w
    })

    # Smoothing
    if smooth_window > 0:
        # Only smooth where we have data?
        # Savgol handles arrays.
        for col in [X, Y, Z]:
            # Simple fill for smoothing
            res_df[col] = savgol_filter(res_df[col], smooth_window, smooth_poly)

    return res_df
