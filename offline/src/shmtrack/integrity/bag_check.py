# shmtrack/integrity/bag_check.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from shmtrack.io.rosbag_reader import RosbagReader


@dataclass
class StreamStats:
    cam: str
    topic: str
    n_frames: int
    t0: float
    t1: float
    duration: float
    fps_mean: float
    fps_median: float
    dt_min: float
    dt_max: float
    dt_std: float
    n_gaps: int
    expected_fps: float
    expected_frames: int
    missing_frames_est: int


def _compute_stats(times: np.ndarray, expected_fps: float, gap_thresh_sec: float) -> Tuple[float, float, float, float, float, int]:
    if len(times) < 2:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0)
    dt = np.diff(times)
    # gaps: significantly larger than nominal dt
    n_gaps = int(np.sum(dt > gap_thresh_sec))
    fps = 1.0 / dt
    return (
        float(np.mean(fps)),
        float(np.median(fps)),
        float(np.min(dt)),
        float(np.max(dt)),
        float(np.std(dt)),
        n_gaps,
    )


def check_bag(
    bag_path: str,
    cams: List[str],
    expected_fps: float = 60.0,
    use_header_stamp: bool = True,
    gap_factor: float = 2.0,
    max_frames_per_cam: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Returns:
      - stats_df: per-camera summary stats
      - times_by_cam: dict cam -> time array used for stats (t_sec)
    """
    r = RosbagReader(bag_path, cams=cams, prefer_compressed=True, strict=False)

    stats: List[StreamStats] = []
    times_by_cam: Dict[str, np.ndarray] = {}

    nominal_dt = 1.0 / expected_fps
    gap_thresh = gap_factor * nominal_dt

    for cam in cams:
        topic = r.topics_for_cam(cam)["image"]
        ts: List[float] = []

        for fr in r.iter_frames(cam, use_header_stamp=use_header_stamp, max_frames=max_frames_per_cam):
            # Prefer header time if available; else fall back to bag time
            t = fr.t_hdr_sec if (use_header_stamp and fr.t_hdr_sec is not None) else fr.t_bag_sec
            ts.append(float(t))

        if len(ts) == 0:
            times_by_cam[cam] = np.array([], dtype=float)
            stats.append(StreamStats(
                cam=cam, topic=topic, n_frames=0,
                t0=0.0, t1=0.0, duration=0.0,
                fps_mean=0.0, fps_median=0.0,
                dt_min=0.0, dt_max=0.0, dt_std=0.0,
                n_gaps=0,
                expected_fps=expected_fps,
                expected_frames=0,
                missing_frames_est=0,
            ))
            continue

        times = np.array(ts, dtype=float)

        # If header timestamps go slightly negative (as you observed), shift to start at 0
        if np.min(times) < 0.0:
            times = times - np.min(times)

        # Ensure monotonic (occasionally header stamps can jitter)
        times = np.maximum.accumulate(times)

        times_by_cam[cam] = times

        t0 = float(times[0])
        t1 = float(times[-1])
        duration = max(0.0, t1 - t0)

        fps_mean, fps_median, dt_min, dt_max, dt_std, n_gaps = _compute_stats(times, expected_fps, gap_thresh)

        expected_frames = int(round(duration * expected_fps)) + 1
        missing = max(0, expected_frames - len(times))

        stats.append(StreamStats(
            cam=cam, topic=topic, n_frames=int(len(times)),
            t0=t0, t1=t1, duration=duration,
            fps_mean=fps_mean, fps_median=fps_median,
            dt_min=dt_min, dt_max=dt_max, dt_std=dt_std,
            n_gaps=n_gaps,
            expected_fps=expected_fps,
            expected_frames=expected_frames,
            missing_frames_est=int(missing),
        ))

    stats_df = pd.DataFrame([s.__dict__ for s in stats])
    return stats_df, times_by_cam


def estimate_intercamera_skew(times_by_cam: Dict[str, np.ndarray], ref_cam: str) -> pd.DataFrame:
    """
    Simple nearest-neighbor skew estimate vs a reference camera.
    For each ref timestamp, find nearest in other cams.
    """
    ref = times_by_cam.get(ref_cam, np.array([], dtype=float))
    if ref.size == 0:
        return pd.DataFrame()

    rows = []
    for cam, ts in times_by_cam.items():
        if cam == ref_cam or ts.size == 0:
            continue
        # nearest neighbor matching
        idx = np.searchsorted(ts, ref)
        idx = np.clip(idx, 0, len(ts) - 1)

        # compare vs previous neighbor too for better nearest choice
        idx_prev = np.clip(idx - 1, 0, len(ts) - 1)
        d1 = np.abs(ts[idx] - ref)
        d0 = np.abs(ts[idx_prev] - ref)
        best = np.where(d0 < d1, idx_prev, idx)

        skew = ts[best] - ref
        rows.append({
            "cam": cam,
            "ref_cam": ref_cam,
            "skew_mean_ms": float(np.mean(skew) * 1000.0),
            "skew_std_ms": float(np.std(skew) * 1000.0),
            "skew_maxabs_ms": float(np.max(np.abs(skew)) * 1000.0),
            "n_pairs": int(len(skew)),
        })

    return pd.DataFrame(rows)
