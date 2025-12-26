# shmtrack/io/schema.py
"""
Canonical schema for shm-vibration-tracking.

Goal:
- Use meaningful column names internally (canonical v2).
- Allow lossless export to legacy CSV schemas for compatibility with old scripts/papers.

Conventions:
- Time: seconds as float (relative to run start): t_sec
- Units: meters for translations, pixels for image-space quantities
- Quaternion: (qx, qy, qz, qw), xyzw order, right-handed
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


# -----------------------
# Canonical (v2) columns
# -----------------------
T = "t_sec"                 # float seconds from run start
CAM = "cam_id"              # "sony_cam1" etc.
TAG = "tag_id"              # int

# Pose (camera or world depending on stage)
X = "x_m"
Y = "y_m"
Z = "z_m"
QX, QY, QZ, QW = "qx", "qy", "qz", "qw"

# AprilTag quality
DECISION_MARGIN = "decision_margin"
HAMMING = "hamming"
TAG_AREA = "tag_area_px2"

# Detection state
DETECTED = "detected"       # 1/0
SOURCE = "source"           # "tag" | "klt" | "fused" | "interp" | "none"

# Optional geometry quality
REPROJ_ERR = "reproj_err_px"

# Paper 2 robustness signals
KLT_NUM = "klt_num_pts"
KLT_RMSE = "klt_rmse_px"
FLOW_MAG = "flow_mag_px"
MCI = "mci"
W_TAG = "w_tag"
W_KLT = "w_klt"

# Common canonical pose record (superset)
POSE_CANONICAL_COLUMNS: List[str] = [
    T, CAM, TAG,
    X, Y, Z,
    QX, QY, QZ, QW,
    DECISION_MARGIN, HAMMING, TAG_AREA,
    DETECTED, SOURCE,
    REPROJ_ERR,
    KLT_NUM, KLT_RMSE,
    FLOW_MAG, MCI,
    W_TAG, W_KLT,
]

POSE_CANONICAL_MIN_COLUMNS: List[str] = [
    T, CAM, TAG,
    X, Y, Z,
    QX, QY, QZ, QW,
]

# World pose uses same fields but interpretations change (world frame)
POSE_WORLD_COLUMNS: List[str] = POSE_CANONICAL_COLUMNS.copy()


# -----------------------
# Legacy mapping support
# -----------------------
# Legacy detect_apriltag.py columns:
# timestamp, tx_m, ty_m, tz_m, qx, qy, qz, qw, tag_id, dm, hamming, area_px2

LEGACY_TO_CANONICAL: Dict[str, str] = {
    "timestamp": T,
    "tx_m": X,
    "ty_m": Y,
    "tz_m": Z,
    "qx": QX,
    "qy": QY,
    "qz": QZ,
    "qw": QW,
    "tag_id": TAG,
    "dm": DECISION_MARGIN,
    "decision_margin": DECISION_MARGIN,
    "hamming": HAMMING,
    "area_px2": TAG_AREA,
    "tag_area_px2": TAG_AREA,
}

CANONICAL_TO_LEGACY: Dict[str, str] = {v: k for k, v in LEGACY_TO_CANONICAL.items()}

LEGACY_DETECT_COLUMNS: List[str] = [
    "timestamp",
    "tx_m", "ty_m", "tz_m",
    "qx", "qy", "qz", "qw",
    "tag_id",
    "dm",
    "hamming",
    "area_px2",
]


def rename_legacy_to_canonical(df):
    cols = {c: LEGACY_TO_CANONICAL[c] for c in df.columns if c in LEGACY_TO_CANONICAL}
    return df.rename(columns=cols)


def rename_canonical_to_legacy(df):
    cols = {c: CANONICAL_TO_LEGACY[c] for c in df.columns if c in CANONICAL_TO_LEGACY}
    return df.rename(columns=cols)


def ensure_columns(df, cols: List[str], fill_value=""):
    for c in cols:
        if c not in df.columns:
            df[c] = fill_value
    return df


def export_legacy_detect_csv(df, out_csv: str):
    """
    Export legacy-compatible detection CSV (Paper 1 / old plotting scripts).
    """
    df2 = rename_canonical_to_legacy(df.copy())
    df2 = ensure_columns(df2, LEGACY_DETECT_COLUMNS, fill_value="")
    df2 = df2[LEGACY_DETECT_COLUMNS]
    df2.to_csv(out_csv, index=False)
