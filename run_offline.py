"""
Offline CoM pipeline (Phase 1): video in -> annotated video + per-frame CSV.

Usage:
    python run_offline.py INPUT.mp4 --table deleva --sex male --view side
    python run_offline.py INPUT.mp4 --table dempster            # swap the BSP table
    python run_offline.py INPUT.mp4 --target                    # draw a stability circle

    # calibrated: undistort the lens, and report the CoM in centimetres
    python run_offline.py INPUT.mp4 --intrinsics camera_1280x720.npz --scale scale.json

Outputs (next to the input, or --outdir):
    INPUT.annotated.mp4   skeleton + CoM marker + sway trail (+ optional target)
    INPUT.com.csv         frame, time_s, CoM (norm, px, and cm if --scale),
                          mass_used, proc_ms
"""

from __future__ import annotations
import argparse
import csv
import json
import os
import time
from collections import deque

import cv2
import numpy as np

from bsp_tables import get_table
from com import compute_com
from pose_estimator import PoseEstimator
from visualize import draw_skeleton, draw_com, put_hud


def parse_args():
    ap = argparse.ArgumentParser(description="Offline markerless CoM pipeline")
    ap.add_argument("input", help="input video path")
    ap.add_argument("--table", default="dempster", choices=["deleva", "dempster"],
                    help="BSP table. dempster is this study's primary; deleva is "
                         "the sensitivity check (and the only sex-specific one).")
    ap.add_argument("--sex", default="male", choices=["male", "female"],
                    help="only used by --table deleva; Dempster has no sex variants")
    ap.add_argument("--view", default="side", choices=["front", "side"],
                    help="labels the sway axis (front=medio-lateral, side=antero-posterior)")
    ap.add_argument("--target", action="store_true",
                    help="draw a stability target centred on the mean CoM (demo biofeedback)")
    ap.add_argument("--target-frac", type=float, default=0.05,
                    help="target radius as a fraction of frame width")
    ap.add_argument("--model-complexity", type=int, default=1, choices=[0, 1, 2])
    ap.add_argument("--trail", type=int, default=48, help="sway-trail length in frames")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--no-video", action="store_true", help="write CSV only (faster)")
    ap.add_argument("--intrinsics", default=None,
                    help="camera_WxH.npz from calibrate_camera.py; removes lens "
                         "distortion before the CoM is computed")
    ap.add_argument("--scale", default=None,
                    help="scale.json from scale_from_board.py; adds com_*_cm "
                         "columns. Requires --intrinsics.")
    args = ap.parse_args()
    if args.scale and not args.intrinsics:
        ap.error("--scale needs --intrinsics (the cm conversion is anchored on "
                 "the principal point cx, cy)")
    return args


def main():
    args = parse_args()
    table = get_table(args.table, sex=args.sex)
    sway_axis = "medio-lateral (L-R)" if args.view == "front" else "antero-posterior (F-B)"
    # Dempster has no sex variants, so don't let a "dempster/male" label imply
    # a sex-specific table in the CSV, the HUD or any figure.
    table_label = f"{args.table}/{args.sex}" if args.table == "deleva" else args.table
    print(f"BSP table     : {table_label}"
          + ("  (--sex ignored: Dempster has no sex variants)"
             if args.table == "dempster" else ""))

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or -1

    # --- optional calibration ------------------------------------------------
    # Landmarks are undistorted before the CoM is computed. The annotated frame
    # is undistorted too, so image, skeleton and marker all agree. (The live
    # loop should undistort POINTS only -- 33 points is far cheaper than a
    # full-frame remap, and the displayed frame does not need to be metric.)
    K = dist = maps = None
    cm_per_px = None
    if args.intrinsics:
        calib = np.load(args.intrinsics)
        K, dist = calib["K"], calib["dist"]
        csize = tuple(int(v) for v in calib["image_size"])
        if csize != (w, h):
            raise SystemExit(
                f"{args.input} is {w}x{h} but {args.intrinsics} was fitted at "
                f"{csize[0]}x{csize[1]}. Intrinsics are resolution-specific: "
                "re-record at the calibrated resolution, or recalibrate."
            )
        maps = cv2.initUndistortRectifyMap(K, dist, None, K, (w, h), cv2.CV_16SC2)
        print(f"Undistorting with {args.intrinsics} "
              f"(RMS {float(calib['rms']):.3f} px)")
    if args.scale:
        with open(args.scale) as f:
            sc = json.load(f)
        if tuple(sc["image_size"]) != (w, h):
            raise SystemExit(
                f"{args.scale} was measured at {sc['image_size']} but this clip "
                f"is {w}x{h}."
            )
        cm_per_px = (float(sc["cm_per_px_x"]), float(sc["cm_per_px_y"]))
        print(f"Metric scale: {cm_per_px[0]:.5f} / {cm_per_px[1]:.5f} cm/px "
              f"at {float(sc['plane_depth_cm']):.0f} cm plane depth")
        if sc.get("session"):
            print(f"  session   : {sc['session']}"
                  + (f"  ({sc['note']})" if sc.get("note") else ""))
        # Z is per-session; the intrinsics are not. Pairing this session's scale
        # with a stale intrinsics file is the realistic mistake here.
        if sc.get("intrinsics") and sc["intrinsics"] != os.path.basename(args.intrinsics):
            print(f"  WARNING: this scale was derived from {sc['intrinsics']} but "
                  f"--intrinsics is {os.path.basename(args.intrinsics)}.")

    base = os.path.splitext(os.path.basename(args.input))[0]
    outdir = args.outdir or os.path.dirname(os.path.abspath(args.input))
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, f"{base}.com.csv")
    vid_path = os.path.join(outdir, f"{base}.annotated.mp4")

    writer = None
    if not args.no_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(vid_path, fourcc, fps, (w, h))

    trail = deque(maxlen=args.trail)
    target_px = None
    target_radius = args.target_frac * w

    proc_times = []
    rows = []
    frame_idx = 0

    with PoseEstimator(model_complexity=args.model_complexity) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if maps is not None:
                frame = cv2.remap(frame, maps[0], maps[1], cv2.INTER_LINEAR)
            t0 = time.perf_counter()
            kps_norm, kps_px, vis = pose(frame)
            com_px = com_cm = None
            info = {}
            if kps_norm is not None:
                com_px, info = compute_com(kps_px, table, visibility=vis)
                if com_px is not None:
                    com_norm = com_px / np.array([w, h])
                    if cm_per_px is not None:
                        # Exact pinhole relation: X = (u - cx) * Z / fx, so the
                        # cm coordinate is an offset from the optical axis in the
                        # subject's plane -- not just a rescaled raw pixel.
                        com_cm = np.array([
                            (com_px[0] - K[0, 2]) * cm_per_px[0],
                            (com_px[1] - K[1, 2]) * cm_per_px[1],
                        ])
            proc_ms = (time.perf_counter() - t0) * 1000.0
            proc_times.append(proc_ms)

            t_s = frame_idx / fps
            if com_px is not None:
                trail.append(com_px)
                if args.target and target_px is None and frame_idx >= 5:
                    target_px = com_px.copy()  # centre target on early CoM (demo)
                rows.append([
                    frame_idx, f"{t_s:.4f}",
                    f"{com_norm[0]:.6f}", f"{com_norm[1]:.6f}",
                    f"{com_px[0]:.3f}", f"{com_px[1]:.3f}",
                    f"{com_cm[0]:.4f}" if com_cm is not None else "",
                    f"{com_cm[1]:.4f}" if com_cm is not None else "",
                    f"{info.get('mass_used', 0):.4f}", f"{proc_ms:.3f}",
                ])
            else:
                rows.append([frame_idx, f"{t_s:.4f}",
                             "", "", "", "", "", "", "", f"{proc_ms:.3f}"])

            if writer is not None:
                if kps_px is not None:
                    draw_skeleton(frame, kps_px, vis)
                if com_px is not None:
                    draw_com(frame, com_px, trail=trail,
                             target_px=target_px if args.target else None,
                             target_radius_px=target_radius if args.target else None)
                hud = [
                    f"table={table_label}  view={args.view}",
                    f"sway axis: {sway_axis}",
                    f"proc {proc_ms:.1f} ms  ({1000.0/max(proc_ms,1e-6):.0f} eff. fps)",
                ]
                if cm_per_px is not None:
                    hud.append(f"scale {cm_per_px[0]:.4f} cm/px"
                               + (f"   CoM {com_cm[0]:+.1f} cm"
                                  if com_cm is not None else ""))
                elif maps is not None:
                    hud.append("undistorted (no metric scale)")
                put_hud(frame, hud)
                writer.write(frame)

            frame_idx += 1
            if n_frames > 0 and frame_idx % 30 == 0:
                print(f"  {frame_idx}/{n_frames} frames", end="\r")

    cap.release()
    if writer is not None:
        writer.release()

    with open(csv_path, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["frame", "time_s", "com_x_norm", "com_y_norm",
                       "com_x_px", "com_y_px", "com_x_cm", "com_y_cm",
                       "mass_used", "proc_ms"])
        wcsv.writerows(rows)

    pt = np.array(proc_times)
    detected = sum(1 for r in rows if r[2] != "")
    print()
    print(f"Frames        : {frame_idx}  ({detected} with CoM, "
          f"{100*detected/max(frame_idx,1):.0f}%)")
    print(f"Source FPS    : {fps:.1f}")
    print(f"Proc time     : mean {pt.mean():.1f} ms  p95 {np.percentile(pt,95):.1f} ms  "
          f"-> {1000.0/pt.mean():.0f} eff. fps (offline, not the live-loop latency)")
    print(f"CSV           : {csv_path}")
    if writer is not None:
        print(f"Annotated vid : {vid_path}")


if __name__ == "__main__":
    main()
