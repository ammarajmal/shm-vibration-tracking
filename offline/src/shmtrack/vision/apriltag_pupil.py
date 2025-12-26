# shmtrack/vision/apriltag_pupil.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
from pupil_apriltags import Detector


@dataclass
class TagDetection:
    tag_id: int
    corners_px: np.ndarray        # (4,2) float32
    center_px: np.ndarray         # (2,) float32
    decision_margin: float
    hamming: int
    tag_area_px2: float


_DET_CACHE: Dict[str, Detector] = {}


def _detector_key(family: str, params: Dict) -> str:
    parts = [family] + [f"{k}={params[k]}" for k in sorted(params.keys())]
    return "|".join(parts)


def get_detector(
    family: str = "tag36h11",
    params: Optional[Dict] = None,
) -> Detector:
    params = params or {}
    key = _detector_key(family, params)
    if key in _DET_CACHE:
        return _DET_CACHE[key]

    det = Detector(
        families=family,
        nthreads=int(params.get("nthreads", 4)),
        quad_decimate=float(params.get("quad_decimate", 1.0)),
        quad_sigma=float(params.get("quad_sigma", 0.8)),
        refine_edges=bool(params.get("refine_edges", True)),
        decode_sharpening=float(params.get("decode_sharpening", 0.25)),
        debug=bool(params.get("debug", False)),
    )
    _DET_CACHE[key] = det
    return det


def detect_tags(
    img_bgr: np.ndarray,
    family: str = "tag36h11",
    params: Optional[Dict] = None,
) -> List[TagDetection]:
    det = get_detector(family=family, params=params)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    ds = det.detect(gray, estimate_tag_pose=False)

    out: List[TagDetection] = []
    for d in ds:
        corners = np.array(d.corners, dtype=np.float32)  # (4,2)
        center = np.array(d.center, dtype=np.float32)
        area = float(cv2.contourArea(corners.reshape(-1, 1, 2)))

        out.append(
            TagDetection(
                tag_id=int(d.tag_id),
                corners_px=corners,
                center_px=center,
                decision_margin=float(getattr(d, "decision_margin", 0.0)),
                hamming=int(getattr(d, "hamming", -1)),
                tag_area_px2=area,
            )
        )
    return out


def pick_best_single_tag(
    detections: List[TagDetection],
    *,
    tag_id: Optional[int] = None,
) -> Optional[TagDetection]:
    """
    WTT default: one tag per camera.
    If tag_id is given, return that tag if present.
    Else choose highest decision_margin, tie-break by area.
    """
    if not detections:
        return None

    if tag_id is not None:
        for d in detections:
            if d.tag_id == tag_id:
                return d
        return None

    # best by decision margin, then area
    detections = sorted(
        detections,
        key=lambda d: (d.decision_margin, d.tag_area_px2),
        reverse=True,
    )
    return detections[0]
