"""
Motion Consistency Index (MCI) calculation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from shmtrack.io.schema import (
    X, Y, Z, FLOW_MAG, MCI, T,
    ensure_columns
)

def compute_mci(df: pd.DataFrame, dt: float = 1/60.0) -> pd.DataFrame:
    """
    Appends 'v_pose', 'v_flow_scaled', and 'mci' to the DataFrame.

    MCI = |V_pose - V_flow| / max(V_pose, V_flow)

    High MCI = Disagreement (Outlier/Drift)
    Low MCI = Agreement (Valid Motion)

    Args:
        df: Input dataframe with pose columns (x_m, y_m, z_m) and flow_mag_px.
        dt: Time step for velocity calculation. If None, derived from T column.

    Returns:
        DataFrame with new columns.
    """
    df = df.copy()

    # Ensure T is sorted
    if T in df.columns:
        df = df.sort_values(T)
        # We could use actual dt from timestamp if dt is not constant
        # but usually fixed dt is assumed for synced data.

    # 1. Calculate Pose Velocity (3D Euclidean speed)
    # Fillna(0) for the first element
    cols = [c for c in [X, Y, Z] if c in df.columns]
    if not cols:
        # Fallback or error?
        # If no pose, MCI is undefined.
        df['v_pose'] = 0.0
        df['v_flow_scaled'] = 0.0
        df[MCI] = 0.0
        return df

    pos_diff = df[cols].diff().fillna(0.0)

    # Distance moved per step
    dist = np.linalg.norm(pos_diff.values, axis=1)

    # If we have variable timestamps
    if T in df.columns:
        t_diff = df[T].diff().fillna(dt)
        # Avoid div by zero
        t_diff[t_diff <= 0] = dt
        v_pose = dist / t_diff.values
    else:
        v_pose = dist / dt

    # 2. Scale Flow Magnitude to Metric Units
    # flow_mag_px is in px/frame (usually).
    # If we want to compare to m/s, we need a scalar.
    # We compute a global alpha based on mean ratio.
    # Filter out static parts to get better ratio?
    # Or just global mean.

    if FLOW_MAG not in df.columns:
         df[FLOW_MAG] = 0.0

    flow_vals = df[FLOW_MAG].fillna(0.0).values

    # Convert flow to per-second if it's per-frame?
    # Usually flow is px/frame.
    # v_pose is m/s.
    # We want to map px/frame -> m/s.
    # flow_m_s = flow_px_frame * (fps) * (m/px)
    # alpha includes (fps * m/px).

    # Avoid divide by zero if flow is all zero
    mean_flow = np.mean(flow_vals)
    if mean_flow < 1e-6:
        alpha = 0.0
    else:
        # Simple ratio of means
        alpha = np.mean(v_pose) / mean_flow

    v_flow_scaled = flow_vals * alpha

    # 3. Compute MCI
    # Add epsilon
    numerator = np.abs(v_pose - v_flow_scaled)
    denominator = np.maximum(v_pose, v_flow_scaled) + 1e-6
    mci = numerator / denominator

    # Update DataFrame
    df['v_pose'] = v_pose
    df['v_flow_scaled'] = v_flow_scaled
    df[MCI] = mci

    return df
