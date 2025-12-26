#!/usr/bin/env python3
import argparse
import os

from shmtrack.integrity.bag_check import check_bag, estimate_intercamera_skew


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--cams", default="sony_cam1,sony_cam2,sony_cam3")
    ap.add_argument("--expected-fps", type=float, default=60.0)
    ap.add_argument("--use-header-stamp", action="store_true")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--ref-cam", default="sony_cam1")
    args = ap.parse_args()

    cams = [c.strip() for c in args.cams.split(",") if c.strip()]
    os.makedirs(args.out_dir, exist_ok=True)

    stats_df, times_by_cam = check_bag(
        bag_path=args.bag,
        cams=cams,
        expected_fps=args.expected_fps,
        use_header_stamp=args.use_header_stamp,
    )

    stats_csv = os.path.join(args.out_dir, "bag_check_stats.csv")
    stats_df.to_csv(stats_csv, index=False)
    print(f"Wrote: {stats_csv}")
    print(stats_df)

    skew_df = estimate_intercamera_skew(times_by_cam, ref_cam=args.ref_cam)
    if len(skew_df) > 0:
        skew_csv = os.path.join(args.out_dir, "bag_check_skew.csv")
        skew_df.to_csv(skew_csv, index=False)
        print(f"Wrote: {skew_csv}")
        print(skew_df)


if __name__ == "__main__":
    main()
