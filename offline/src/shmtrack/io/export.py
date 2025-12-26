# shmtrack/io/export.py
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import csv
import yaml
import cv2

from shmtrack.io.rosbag_reader import RosbagReader, CameraInfoData


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_camera_info_yaml(caminfo: CameraInfoData, out_path: Path) -> None:
    data = {
        "K": caminfo.K.tolist(),
        "D": caminfo.D.tolist(),
        "width": int(caminfo.width),
        "height": int(caminfo.height),
        "frame_id": str(caminfo.frame_id),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def export_bag_to_frames(
    bag_path: str,
    cams: List[str],
    out_root: str,
    *,
    prefer_compressed: bool = True,
    use_header_stamp: bool = True,
    normalize_time_to_bag_start: bool = True,
    write_png: bool = True,
    png_compression: int = 3,
    max_frames_per_cam: Optional[int] = None,
) -> Dict[str, Path]:
    """
    Export frames + timestamps + camera_info for each camera.

    Output layout:
      out_root/
        export/<cam>/
          frames/000001.png ...
          frames.csv
          camera_info.yaml

    Returns dict cam -> cam_export_dir
    """
    out_root_p = Path(out_root).expanduser().resolve()
    export_root = out_root_p / "export"
    _safe_mkdir(export_root)

    r = RosbagReader(
        bag_path=bag_path,
        cams=cams,
        prefer_compressed=prefer_compressed,
        strict=False,
    )

    cam_dirs: Dict[str, Path] = {}

    for cam in cams:
        cam_dir = export_root / cam
        frames_dir = cam_dir / "frames"
        _safe_mkdir(frames_dir)
        cam_dirs[cam] = cam_dir

        # Write camera_info once
        caminfo = r.get_camera_info(cam)
        if caminfo is not None:
            _write_camera_info_yaml(caminfo, cam_dir / "camera_info.yaml")

        # Write frames.csv
        csv_path = cam_dir / "frames.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "frame_idx",
                    "t_sec",
                    "t_hdr_sec",
                    "t_bag_sec",
                    "image_path",
                ],
            )
            writer.writeheader()

            idx = 0
            for fr in r.iter_frames(
                cam,
                use_header_stamp=use_header_stamp,
                normalize_time_to_bag_start=normalize_time_to_bag_start,
                max_frames=max_frames_per_cam,
            ):
                idx += 1

                # Choose the “analysis time base”
                t_sec = fr.t_hdr_sec if (use_header_stamp and fr.t_hdr_sec is not None) else fr.t_bag_sec

                # Save image
                if write_png:
                    img_name = f"{idx:06d}.png"
                    out_img = frames_dir / img_name
                    # PNG compression: 0 (best quality, huge) .. 9 (small, slower)
                    cv2.imwrite(str(out_img), fr.image_bgr, [cv2.IMWRITE_PNG_COMPRESSION, int(png_compression)])
                else:
                    img_name = f"{idx:06d}.jpg"
                    out_img = frames_dir / img_name
                    cv2.imwrite(str(out_img), fr.image_bgr)

                writer.writerow(
                    {
                        "frame_idx": idx,
                        "t_sec": float(t_sec),
                        "t_hdr_sec": "" if fr.t_hdr_sec is None else float(fr.t_hdr_sec),
                        "t_bag_sec": float(fr.t_bag_sec),
                        "image_path": str(out_img),
                    }
                )

    return cam_dirs
