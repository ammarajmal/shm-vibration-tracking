# offline/src/shmtrack/vision/klt_fallback.py
import cv2
import numpy as np
from typing import Tuple, Optional, Any

def track_corners_lk(
    img_prev: np.ndarray,
    img_curr: np.ndarray,
    corners_prev: np.ndarray,
    lk_params: Optional[dict] = None
) -> Tuple[np.ndarray, bool, float]:
    """
    Track 4 tag corners from prev to curr using Lucas-Kanade.

    Args:
        img_prev: Previous grayscale image.
        img_curr: Current grayscale image.
        corners_prev: (4, 2) array of previous corner coordinates.
        lk_params: Dictionary of parameters for cv2.calcOpticalFlowPyrLK.

    Returns:
        (new_corners, success_flag, error_metric)
        new_corners: (4, 2) array of tracked corners.
        success_flag: True if tracking was successful and geometrically valid.
        error_metric: Mean forward-backward error or tracking error.
    """
    if lk_params is None:
        lk_params = dict(winSize=(21, 21), maxLevel=3,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    # Ensure images are grayscale
    if len(img_prev.shape) == 3:
        img_prev_gray = cv2.cvtColor(img_prev, cv2.COLOR_BGR2GRAY)
    else:
        img_prev_gray = img_prev

    if len(img_curr.shape) == 3:
        img_curr_gray = cv2.cvtColor(img_curr, cv2.COLOR_BGR2GRAY)
    else:
        img_curr_gray = img_curr

    # 1. Forward Tracking
    p0 = corners_prev.astype(np.float32).reshape(-1, 1, 2)
    p1, st, err = cv2.calcOpticalFlowPyrLK(img_prev_gray, img_curr_gray, p0, None, **lk_params)

    # 2. Check tracking status
    if st is None or np.sum(st) < 4:
        return np.zeros_like(corners_prev), False, 999.0

    p1 = p1.reshape(4, 2)

    # 3. Geometric integrity check
    # Check if area changed drastically.
    # AprilTag corners are ordered counter-clockwise usually? 0->1->2->3
    # Use cv2.contourArea

    area_prev = cv2.contourArea(corners_prev.astype(np.float32))
    area_curr = cv2.contourArea(p1)

    if area_prev <= 1e-6: # Avoid division by zero
        return np.zeros_like(corners_prev), False, 999.0

    area_ratio = area_curr / area_prev

    # Allow some scale change (zoom/Z-movement), but collapse or explode is bad.
    # Say 0.5 to 2.0 range.
    if not (0.5 <= area_ratio <= 2.0):
        return p1, False, 999.0

    # Also check convexity?
    if not cv2.isContourConvex(p1.astype(np.float32).reshape(4, 1, 2)):
         return p1, False, 999.0

    # Return mean error from LK
    mean_err = np.mean(err) if err is not None else 0.0

    return p1, True, float(mean_err)
