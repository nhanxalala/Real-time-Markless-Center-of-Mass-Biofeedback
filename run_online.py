"""
Online CoM pipeline (feasibility demo): live camera or file in -> on-screen
overlay + real-time latency stats. Same pose/CoM/draw stages as
run_offline.py, so the two are numerically comparable; this script measures
live loop latency instead of writing an archival annotated video by default.

Usage:
    python run_online.py                          # webcam 0, dempster table
    python run_online.py --camera 1               # a different webcam
    python run_online.py --input my_side_clip.mp4 # rehearse against a file
    python run_online.py --record                 # also write live.annotated.mp4 / live.com.csv

Quit with 'q' or Esc.
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
    ap = argparse.ArgumentParser(description="Online (live) markerless CoM feasibility loop")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--camera", type=int, default=None,
                      help="camera index for cv2.VideoCapture (default: 0 if --input not given)")
    src.add_argument("--input", default=None,
                      help="video file path, for rehearsing without a webcam")
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
    ap.add_argument("--intrinsics", default=None,
                    help="camera_WxH.npz from calibrate_camera.py; removes lens "
                         "distortion before the CoM is computed")
    ap.add_argument("--scale", default=None,
                    help="scale.json from scale_from_board.py; adds com_*_cm "
                         "columns. Requires --intrinsics.")
    ap.add_argument("--record", action="store_true",
                    help="also write live.annotated.mp4 and live.com.csv to --outdir")
    ap.add_argument("--outdir", default=".", help="output dir for --record")
    args = ap.parse_args()
    if args.scale and not args.intrinsics:
        ap.error("--scale needs --intrinsics (the cm conversion is anchored on "
                 "the principal point cx, cy)")
    if args.camera is None and args.input is None:
        args.camera = 0
    return args


def _open_capture(args):
    src = args.camera if args.input is None else args.input
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        kind = f"camera {args.camera}" if args.input is None else args.input
        raise SystemExit(f"Could not open {kind}")
    return cap


def main():
    args = parse_args()
    table = get_table(args.table, sex=args.sex)
    sway_axis = "medio-lateral (L-R)" if args.view == "front" else "antero-posterior (F-B)"
    table_label = f"{args.table}/{args.sex}" if args.table == "deleva" else args.table
    print(f"BSP table     : {table_label}"
          + ("  (--sex ignored: Dempster has no sex variants)"
             if args.table == "dempster" else ""))
    is_live = args.input is None

    cap = _open_capture(args)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    K = dist = maps = None
    cm_per_px = None
    if args.intrinsics:
        calib = np.load(args.intrinsics)
        K, dist = calib["K"], calib["dist"]
        csize = tuple(int(v) for v in calib["image_size"])
        if csize != (w, h):
            msg = (f"source is {w}x{h} but {args.intrinsics} was fitted at "
                   f"{csize[0]}x{csize[1]}.")
            if is_live:
                print(f"WARNING: {msg} Skipping undistort (webcams can "
                      "misreport size on the first read).")
            else:
                raise SystemExit(f"{msg} Intrinsics are resolution-specific: "
                                  "re-record at the calibrated resolution, or recalibrate.")
        else:
            maps = cv2.initUndistortRectifyMap(K, dist, None, K, (w, h), cv2.CV_16SC2)
            print(f"Undistorting with {args.intrinsics} "
                  f"(RMS {float(calib['rms']):.3f} px)")
    if args.scale:
        with open(args.scale) as f:
            sc = json.load(f)
        if tuple(sc["image_size"]) != (w, h):
            msg = f"{args.scale} was measured at {sc['image_size']} but the source is {w}x{h}."
            if is_live:
                print(f"WARNING: {msg} Skipping metric scale.")
            else:
                raise SystemExit(msg)
        else:
            cm_per_px = (float(sc["cm_per_px_x"]), float(sc["cm_per_px_y"]))
            print(f"Metric scale: {cm_per_px[0]:.5f} / {cm_per_px[1]:.5f} cm/px "
                  f"at {float(sc['plane_depth_cm']):.0f} cm plane depth")
            if sc.get("session"):
                print(f"  session   : {sc['session']}"
                      + (f"  ({sc['note']})" if sc.get("note") else ""))
            # Z is per-session; the intrinsics are not. Pairing this session's
            # scale with a stale intrinsics file is the realistic mistake here.
            if sc.get("intrinsics") and sc["intrinsics"] != os.path.basename(args.intrinsics):
                print(f"  WARNING: this scale was derived from {sc['intrinsics']} but "
                      f"--intrinsics is {os.path.basename(args.intrinsics)}.")

    trail = deque(maxlen=args.trail)
    target_px = None
    target_radius = args.target_frac * w

    win_cap, win_pose, win_com, win_total = (deque(maxlen=30) for _ in range(4))
    all_cap, all_pose, all_com, all_total = [], [], [], []
    rows = []
    writer = None
    frame_idx = 0
    detected = 0

    with PoseEstimator(model_complexity=args.model_complexity) as pose:
        while True:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            t_cap = (time.perf_counter() - t0) * 1000.0
            if not ok:
                if is_live:
                    continue
                break

            if maps is not None:
                frame = cv2.remap(frame, maps[0], maps[1], cv2.INTER_LINEAR)

            t1 = time.perf_counter()
            kps_norm, kps_px, vis = pose(frame)
            t_pose = (time.perf_counter() - t1) * 1000.0

            t2 = time.perf_counter()
            com_px = com_cm = None
            info = {}
            if kps_norm is not None:
                com_px, info = compute_com(kps_px, table, visibility=vis)
                if com_px is not None:
                    com_norm = com_px / np.array([w, h])
                    if cm_per_px is not None:
                        com_cm = np.array([
                            (com_px[0] - K[0, 2]) * cm_per_px[0],
                            (com_px[1] - K[1, 2]) * cm_per_px[1],
                        ])

            if com_px is not None:
                detected += 1
                trail.append(com_px)
                if args.target and target_px is None and frame_idx >= 5:
                    target_px = com_px.copy()
                if kps_px is not None:
                    draw_skeleton(frame, kps_px, vis)
                draw_com(frame, com_px, trail=trail,
                         target_px=target_px if args.target else None,
                         target_radius_px=target_radius if args.target else None)

            hud = [
                f"table={table_label}  view={args.view}  source={'camera' if is_live else args.input}",
                f"sway axis: {sway_axis}",
            ]
            if cm_per_px is not None and com_cm is not None:
                hud.append(f"scale {cm_per_px[0]:.4f} cm/px   CoM {com_cm[0]:+.1f} cm")
            elif maps is not None:
                hud.append("undistorted (no metric scale)")

            t_com = (time.perf_counter() - t2) * 1000.0
            t_total = t_cap + t_pose + t_com

            for w_, v in ((win_cap, t_cap), (win_pose, t_pose), (win_com, t_com), (win_total, t_total)):
                w_.append(v)
            hud.append(
                f"cap {np.mean(win_cap):.1f}  pose {np.mean(win_pose):.1f}  "
                f"com+draw {np.mean(win_com):.1f}  |  total {np.mean(win_total):.1f} ms "
                f"({1000.0/max(np.mean(win_total),1e-6):.0f} fps, smoothed/30)"
            )
            put_hud(frame, hud)
            cv2.imshow("CoM live", frame)

            all_cap.append(t_cap); all_pose.append(t_pose)
            all_com.append(t_com); all_total.append(t_total)

            t_s = frame_idx / max(1000.0 / max(np.mean(win_total), 1e-6), 1e-6)
            if args.record:
                if com_px is not None:
                    rows.append([frame_idx, f"{t_s:.4f}",
                                 f"{com_norm[0]:.6f}", f"{com_norm[1]:.6f}",
                                 f"{com_px[0]:.3f}", f"{com_px[1]:.3f}",
                                 f"{com_cm[0]:.4f}" if com_cm is not None else "",
                                 f"{com_cm[1]:.4f}" if com_cm is not None else "",
                                 f"{info.get('mass_used', 0):.4f}",
                                 f"{t_cap:.3f}", f"{t_pose:.3f}", f"{t_com:.3f}"])
                else:
                    rows.append([frame_idx, f"{t_s:.4f}", "", "", "", "", "", "", "",
                                 f"{t_cap:.3f}", f"{t_pose:.3f}", f"{t_com:.3f}"])
                if writer is None and frame_idx >= 30:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    rec_fps = 1000.0 / max(np.mean(all_total[-30:]), 1e-6)
                    os.makedirs(args.outdir, exist_ok=True)
                    writer = cv2.VideoWriter(os.path.join(args.outdir, "live.annotated.mp4"),
                                              fourcc, rec_fps, (w, h))
                if writer is not None:
                    writer.write(frame)

            frame_idx += 1
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    if args.record:
        csv_path = os.path.join(args.outdir, "live.com.csv")
        with open(csv_path, "w", newline="") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(["frame", "time_s", "com_x_norm", "com_y_norm",
                           "com_x_px", "com_y_px", "com_x_cm", "com_y_cm",
                           "mass_used", "cap_ms", "pose_ms", "com_draw_ms"])
            wcsv.writerows(rows)

    def _stats(name, arr):
        a = np.array(arr)
        return f"{name:9s} mean {a.mean():6.1f} ms  p95 {np.percentile(a,95):6.1f} ms"

    print()
    print(f"Frames        : {frame_idx}  ({detected} with CoM, "
          f"{100*detected/max(frame_idx,1):.0f}%)")
    print("Stage latency :")
    print("  " + _stats("cap", all_cap))
    print("  " + _stats("pose", all_pose))
    print("  " + _stats("com+draw", all_com))
    tot = np.array(all_total)
    print(f"Total latency : mean {tot.mean():.1f} ms  median {np.median(tot):.1f} ms  "
          f"p95 {np.percentile(tot,95):.1f} ms  -> {1000.0/tot.mean():.0f} live fps")
    if args.record:
        print(f"CSV           : {csv_path}")
        print(f"Annotated vid : {os.path.join(args.outdir, 'live.annotated.mp4')}")


if __name__ == "__main__":
    main()
