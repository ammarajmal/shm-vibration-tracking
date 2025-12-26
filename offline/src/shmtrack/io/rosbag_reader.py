#!/usr/bin/env python3
"""
shmtrack.io.rosbag_reader
-------------------------

Robust rosbag reader for multi-camera experiments.

Supports:
- sensor_msgs/Image
- sensor_msgs/CompressedImage
- sensor_msgs/CameraInfo

Key design choices:
- Preserve BOTH time bases:
    * t_bag_sec: rosbag time (connection time)
    * t_hdr_sec: message header stamp (camera/driver time) if available
- Auto-detect raw vs compressed topics
- Streaming iterator (no full bag load into RAM)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

# ROS imports (offline environment still needs rosbag + message definitions)
import rosbag
from cv_bridge import CvBridge

try:
    import cv2  # noqa: F401
except Exception:
    # OpenCV is required for decoding, but keep import error explicit later
    pass


@dataclass(frozen=True)
class CameraInfoData:
    K: np.ndarray        # (3,3)
    D: np.ndarray        # (N,)
    width: int
    height: int
    frame_id: str = ""


@dataclass(frozen=True)
class Frame:
    cam: str
    topic: str
    index: int

    t_bag_sec: float
    t_hdr_sec: Optional[float]

    image_bgr: np.ndarray  # uint8 BGR image


class RosbagReader:
    """
    Read multi-camera image streams from a bag.

    Example:
        r = RosbagReader("run1.bag", cams=["sony_cam1","sony_cam2"])
        caminfo = r.get_camera_info("sony_cam1")
        for fr in r.iter_frames("sony_cam1"):
            ...
    """

    def __init__(
        self,
        bag_path: str | Path,
        cams: List[str],
        prefer_compressed: bool = True,
        strict: bool = False,
    ) -> None:
        self.bag_path = Path(bag_path).expanduser().resolve()
        self.cams = list(cams)
        self.prefer_compressed = prefer_compressed
        self.strict = strict

        if not self.bag_path.exists():
            raise FileNotFoundError(f"Bag not found: {self.bag_path}")

        self._bridge = CvBridge()

        # Cache topics/types once
        with rosbag.Bag(str(self.bag_path)) as bag:
            info = bag.get_type_and_topic_info()
            self._topic_types = {t: meta.msg_type for t, meta in info.topics.items()}
            self._bag_start = float(bag.get_start_time())
            self._bag_end = float(bag.get_end_time())

        # Resolve per-cam topics
        self._img_topics: Dict[str, str] = {}
        self._info_topics: Dict[str, str] = {}

        for cam in self.cams:
            self._info_topics[cam] = f"/{cam}/camera_info"
            self._img_topics[cam] = self._resolve_image_topic(cam)

            if self._img_topics[cam] is None:
                msg = f"No image topic found for cam={cam} in bag={self.bag_path}"
                if self.strict:
                    raise RuntimeError(msg)
                else:
                    # Keep as missing; iter_frames will yield nothing.
                    self._img_topics[cam] = ""

    @property
    def bag_time_range(self) -> Tuple[float, float]:
        """(start_sec, end_sec) of the bag."""
        return self._bag_start, self._bag_end

    def topics_for_cam(self, cam: str) -> Dict[str, str]:
        return {"image": self._img_topics.get(cam, ""), "camera_info": self._info_topics.get(cam, "")}

    def _resolve_image_topic(self, cam: str) -> str:
        """
        Prefer compressed if present:
            /{cam}/image_raw/compressed
        else raw:
            /{cam}/image_raw
        """
        compressed = f"/{cam}/image_raw/compressed"
        raw = f"/{cam}/image_raw"

        has_compressed = compressed in self._topic_types
        has_raw = raw in self._topic_types

        if self.prefer_compressed:
            if has_compressed:
                return compressed
            if has_raw:
                return raw
        else:
            if has_raw:
                return raw
            if has_compressed:
                return compressed

        return ""

    def get_camera_info(self, cam: str) -> Optional[CameraInfoData]:
        """
        Return first CameraInfo found for cam, or None if missing.
        """
        info_topic = self._info_topics.get(cam, "")
        if not info_topic or info_topic not in self._topic_types:
            return None

        with rosbag.Bag(str(self.bag_path)) as bag:
            for _, msg, _ in bag.read_messages(topics=[info_topic]):
                K = np.array(msg.K, dtype=float).reshape(3, 3)
                D = np.array(msg.D, dtype=float)
                return CameraInfoData(
                    K=K,
                    D=D,
                    width=int(getattr(msg, "width", 0)),
                    height=int(getattr(msg, "height", 0)),
                    frame_id=str(getattr(msg.header, "frame_id", "")) if hasattr(msg, "header") else "",
                )
        return None

    def iter_frames(
        self,
        cam: str,
        *,
        use_header_stamp: bool = True,
        normalize_time_to_bag_start: bool = True,
        max_frames: Optional[int] = None,
    ) -> Iterator[Frame]:
        """
        Stream frames for one camera.

        Times returned:
          - t_bag_sec is always available (from rosbag t)
          - t_hdr_sec is msg.header.stamp.to_sec() if present and >0 else None
        """
        topic = self._img_topics.get(cam, "")
        if not topic:
            return iter(())

        bag_base = self._bag_start if normalize_time_to_bag_start else 0.0

        # NEW: per-camera header base (first valid header stamp)
        hdr_base: Optional[float] = None

        bridge = self._bridge
        with rosbag.Bag(str(self.bag_path)) as bag:
            idx = 0
            for _, msg, t in bag.read_messages(topics=[topic]):
                idx += 1
                if max_frames is not None and idx > max_frames:
                    break

                t_bag_sec = float(t.to_sec()) - bag_base

                t_hdr_sec: Optional[float] = None
                if use_header_stamp and hasattr(msg, "header") and hasattr(msg.header, "stamp"):
                    hs = float(msg.header.stamp.to_sec())
                    if hs > 0:
                        if hdr_base is None:
                            hdr_base = hs
                        t_hdr_sec = hs - hdr_base  # always starts at 0.0 for this camera

                # Decode image
                msg_type = self._topic_types.get(topic, "")
                try:
                    if msg_type.endswith("CompressedImage"):
                        img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
                    elif msg_type.endswith("Image"):
                        img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                    else:
                        # Fallback: try to detect by attribute
                        if hasattr(msg, "format") and hasattr(msg, "data"):
                            img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
                        else:
                            img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                except Exception as e:
                    if self.strict:
                        raise
                    # Skip bad frames in non-strict mode
                    continue

                yield Frame(
                    cam=cam,
                    topic=topic,
                    index=idx,
                    t_bag_sec=t_bag_sec,
                    t_hdr_sec=t_hdr_sec,
                    image_bgr=img,
                )
