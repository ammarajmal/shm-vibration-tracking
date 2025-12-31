# offline/src/shmtrack/vision/flow_roi.py
import cv2
import numpy as np
from typing import Optional, Union

def compute_flow_roi(
    img_prev: np.ndarray,
    img_curr: np.ndarray,
    roi_corners: np.ndarray,
    margin_px: int = 20,
    pyr_scale: float = 0.5,
    levels: int = 3,
    winsize: int = 15,
    iterations: int = 3,
    poly_n: int = 5,
    poly_sigma: float = 1.2,
    flags: int = 0
) -> float:
    """
    Calculate average optical flow magnitude within a Region of Interest (ROI).

    Args:
        img_prev: Previous frame.
        img_curr: Current frame.
        roi_corners: (4, 2) array defining the quad of the tag/ROI.
        margin_px: Extra padding around the bounding box of roi_corners.

        ...Farneback flow parameters...

    Returns:
        mean_flow_mag: Average magnitude of flow vectors inside the ROI.
    """
    # 1. Determine Bounding Box
    x_min = int(np.min(roi_corners[:, 0])) - margin_px
    x_max = int(np.max(roi_corners[:, 0])) + margin_px
    y_min = int(np.min(roi_corners[:, 1])) - margin_px
    y_max = int(np.max(roi_corners[:, 1])) + margin_px

    h, w = img_prev.shape[:2]

    # Clip to image bounds
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w, x_max)
    y_max = min(h, y_max)

    if x_max <= x_min or y_max <= y_min:
        return 0.0

    # 2. Crop
    roi_prev = img_prev[y_min:y_max, x_min:x_max]
    roi_curr = img_curr[y_min:y_max, x_min:x_max]

    # Ensure grayscale
    if len(roi_prev.shape) == 3:
        roi_prev = cv2.cvtColor(roi_prev, cv2.COLOR_BGR2GRAY)
    if len(roi_curr.shape) == 3:
        roi_curr = cv2.cvtColor(roi_curr, cv2.COLOR_BGR2GRAY)

    # 3. Compute Dense Flow (Farneback)
    flow = cv2.calcOpticalFlowFarneback(
        roi_prev, roi_curr, None,
        pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags
    )

    # 4. Compute Magnitude
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # 5. Return Mean Magnitude
    return float(np.mean(mag))
