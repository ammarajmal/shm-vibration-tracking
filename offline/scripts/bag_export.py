#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from shmtrack.io.export import export_bag_to_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True, help="Path to .bag")
    ap.add_argument("--cams", default="sony_cam1,sony_cam2,sony_cam3")
    ap.add_argument("--out", required=True, help="Output run directory, e.g. results/e7_90rpm_run1")
    ap.add_argument("--use-header-stamp", action="store_true", help="Use header.stamp as analysis time base")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--png-compression", type=int, default=3, help="0..9 (lower is larger/faster)")
    args = ap.parse_args()

    cams = [c.strip() for c in args.cams.split(",") if c.strip()]
    out_run = Path(args.out).expanduser().resolve()
    out_run.mkdir(parents=True, exist_ok=True)

    export_bag_to_frames(
        bag_path=args.bag,
        cams=cams,
        out_root=str(out_run),
        use_header_stamp=args.use_header_stamp,
        write_png=True,
        png_compression=args.png_compression,
        max_frames_per_cam=args.max_frames,
    )

    print(f"✅ Export complete -> {out_run}/export/<cam>/")


if __name__ == "__main__":
    main()
