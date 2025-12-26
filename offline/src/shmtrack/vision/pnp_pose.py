# shmtrack/vision/pnp_pose.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

from shmtrack.io.rosbag_reader import CameraInfoData


@dataclass(frozen=True)
class Pose:
    x_m: float
    y_m: float
    z_m: float
    qx: float
    qy: float
    qz: float
    qw: float


def tag_object_points(tag_size_m: float) -> np.ndarray:
    """
    Tag corners in tag frame (Z=0). Order must match detector corners order.
    Using standard square corners around origin.
    """
    s = float(tag_size_m)
    h = s / 2.0
    return np.array(
        [
            [-h, -h, 0.0],
            [ h, -h, 0.0],
            [ h,  h, 0.0],
            [-h,  h, 0.0],
        ],
        dtype=np.float32,
    )


def solve_pnp_from_corners(
    corners_px: np.ndarray,     # (4,2)
    cam: CameraInfoData,
    tag_size_m: float,
    use_ransac: bool = False,
) -> Tuple[Pose, float]:
    """
    Returns (pose, reproj_err_px_mean).
    """
    objp = tag_object_points(tag_size_m)
    imgp = corners_px.astype(np.float32)

    K = cam.K.astype(np.float64)
    D = cam.D.astype(np.float64)

    if use_ransac:
        ok, rvec, tvec, _ = cv2.solvePnPRansac(objp, imgp, K, D)
    else:
        ok, rvec, tvec = cv2.solvePnP(objp, imgp, K, D, flags=cv2.SOLVEPNP_ITERATIVE)

    if not ok:
        raise RuntimeError("solvePnP failed")

    # reprojection error
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, D)
    proj = proj.reshape(-1, 2)
    err = float(np.mean(np.linalg.norm(proj - imgp, axis=1)))

    rot = R.from_rotvec(rvec.reshape(3))
    qx, qy, qz, qw = rot.as_quat()  # xyzw

    pose = Pose(
        x_m=float(tvec[0]),
        y_m=float(tvec[1]),
        z_m=float(tvec[2]),
        qx=float(qx), qy=float(qy), qz=float(qz), qw=float(qw),
    )
    return pose, err
