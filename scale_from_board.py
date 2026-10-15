"""
Pixels -> centimetres for the CoM signal.

Intrinsics (calibrate_camera.py) give a ray per pixel, not a length. A pixel
only becomes a centimetre once you know how far away the thing is:

    cm_per_px = Z_cm / fx          (fx in pixels)

...where Z is the depth of the PLANE the motion happens in. That scale is valid
for motion in that plane, which is exactly this study's case: front view ->
medio-lateral sway is in-plane; side view -> antero-posterior sway is in-plane.
The out-of-plane component of each view is the one you do not measure, which is
why both views get recorded.

Two ways to get Z. Pick by how big the board is in frame:

  --distance-cm 300          tape-measured camera-to-subject distance.
                             RECOMMENDED at typical standing distances. The
                             only ambiguity is where the optical centre sits
                             inside the lens (1-2 cm), i.e. <1% at 3 m.

  --board IMG.png            solvePnP on a checkerboard held in the subject's
                             plane at CoM height (navel). More exact, but ONLY
                             if the board spans a useful part of the frame --
                             a 30 cm board at 3 m in a 640 px frame is ~45 px
                             across and the corner fit is meaningless. This
                             mode refuses to answer when the board is too small.

Intrinsics are solved ONCE per camera + resolution. Z is a property of the
SESSION, not the camera: move the tripod, or stand the subject at a different
distance, and Z changed. Tag each measurement with --session so the scale file
that produced a given set of centimetres is identifiable afterwards. Front and
side views are two different plane depths, so they are two sessions' scales.

Usage:
    python scale_from_board.py --intrinsics camera_640x368.npz --distance-cm 300
    python scale_from_board.py --intrinsics camera_640x368.npz \
        --board subject_plane.png --cols 9 --rows 6 --square-mm 60.0

    # per session: writes scale_s01_front.json
    python scale_from_board.py --intrinsics camera_1280x720.npz --distance-cm 280 \
        --session s01_front --note "front view, tripod 2.8 m, subject barefoot"
"""

from __future__ import annotations
import argparse
import datetime
import json
import os

import cv2
import numpy as np

from calibrate_camera import find_corners, object_points

MIN_BOARD_FRAME_FRAC = 0.15   # board must span >=15% of frame width
MIN_SQUARE_PX = 10.0          # cornerSubPix needs squares bigger than its window


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--intrinsics", required=True,
                    help="camera_WxH.npz from calibrate_camera.py")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--distance-cm", type=float,
                    help="tape-measured distance, camera to the subject's plane")
    src.add_argument("--board", help="frame with the board held in the subject's plane")
    ap.add_argument("--cols", type=int, help="INNER corners across (--board only)")
    ap.add_argument("--rows", type=int, help="INNER corners down (--board only)")
    ap.add_argument("--square-mm", type=float, help="square side in mm (--board only)")
    ap.add_argument("--session", default=None,
                    help="session tag, e.g. s01_front. Names the output "
                         "scale_<session>.json and is stamped into it.")
    ap.add_argument("--note", default=None,
                    help="free-text session setup note, e.g. "
                         "'front view, tripod 2.8 m, subject barefoot'")
    ap.add_argument("--out", default=None,
                    help="output path (default scale_<session>.json, or "
                         "scale.json with no --session)")
    args = ap.parse_args()
    if args.board and not all([args.cols, args.rows, args.square_mm]):
        ap.error("--board needs --cols, --rows and --square-mm")
    if args.out is None:
        args.out = f"scale_{args.session}.json" if args.session else "scale.json"
    return args


def z_from_board(args, K, dist, calib_size):
    """solvePnP the board -> (Z_cm, extras dict). Refuses if the board is small."""
    img = cv2.imread(args.board)
    if img is None:
        raise SystemExit(f"Could not read {args.board}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    size = gray.shape[::-1]
    if size != calib_size:
        raise SystemExit(
            f"{args.board} is {size} but the intrinsics were fitted at {calib_size}. "
            "Intrinsics are resolution-specific -- re-shoot this frame at the "
            "calibrated resolution, or recalibrate."
        )

    pattern = (args.cols, args.rows)
    corners = find_corners(gray, pattern)
    if corners is None:
        raise SystemExit("No board found. Check --cols/--rows are INNER corners "
                         "(a 10x7-square board has 9x6), and that the whole board "
                         "is in frame and in focus.")

    c = corners.reshape(-1, 2)
    span_px = float(np.ptp(c[:, 0]))
    frac = span_px / size[0]
    grid = corners.reshape(args.rows, args.cols, 2)
    square_px = float(np.linalg.norm(np.diff(grid, axis=1), axis=2).mean())
    if frac < MIN_BOARD_FRAME_FRAC or square_px < MIN_SQUARE_PX:
        raise SystemExit(
            f"Board is only {span_px:.0f} px across ({100*frac:.0f}% of the frame) "
            f"with {square_px:.1f} px squares -- too small for a trustworthy "
            f"corner fit (need >={100*MIN_BOARD_FRAME_FRAC:.0f}% of frame width and "
            f">={MIN_SQUARE_PX:.0f} px squares).\n"
            "Fix it one of three ways:\n"
            "  1. use --distance-cm with a tape measure instead (fine at <1% error),\n"
            "  2. record at a higher resolution (1280x720 or 1920x1080),\n"
            "  3. print a bigger board (A2/A1, 60-80 mm squares)."
        )

    objp = object_points(args.cols, args.rows, args.square_mm)
    ok, rvec, tvec = cv2.solvePnP(objp, corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise SystemExit("solvePnP failed.")
    Z_cm = float(tvec.ravel()[2]) / 10.0

    # How far off fronto-parallel is the board? The scale is the scale of the
    # board's plane; badly tilted, that plane is not the subject's coronal plane.
    R, _ = cv2.Rodrigues(rvec)
    tilt_deg = float(np.degrees(np.arccos(np.clip(abs(R[2, 2]), -1.0, 1.0))))

    # Model-free cross-check: undistort the corners and measure the mean spacing
    # of adjacent corners along each board row. Only valid near fronto-parallel.
    und = cv2.undistortPoints(corners, K, dist, P=K).reshape(args.rows, args.cols, 2)
    spacing_px = float(np.linalg.norm(np.diff(und, axis=1), axis=2).mean())
    cm_per_px_direct = (args.square_mm / 10.0) / spacing_px

    return Z_cm, {"board_tilt_deg": tilt_deg,
                  "board_frame_frac": frac,
                  "board_square_px": square_px,
                  "cm_per_px_direct": cm_per_px_direct,
                  "image_size": list(size)}


def main():
    args = parse_args()
    calib = np.load(args.intrinsics)
    K, dist = calib["K"], calib["dist"]
    calib_size = tuple(int(v) for v in calib["image_size"])
    fx, fy = float(K[0, 0]), float(K[1, 1])
    rms = float(calib["rms"])

    if args.board:
        Z_cm, extra = z_from_board(args, K, dist, calib_size)
        method = "solvePnP on a checkerboard in the subject's plane"
    else:
        Z_cm, extra = args.distance_cm, {"image_size": list(calib_size)}
        method = "tape-measured camera-to-subject distance"

    cm_per_px_x = Z_cm / fx
    cm_per_px_y = Z_cm / fy
    w, h = extra["image_size"]

    if args.session:
        print(f"\nsession       : {args.session}")
        if args.note:
            print(f"note          : {args.note}")
    print(f"\nintrinsics    : {os.path.basename(args.intrinsics)}  "
          f"fx={fx:.2f} fy={fy:.2f}  (RMS {rms:.3f} px, {calib_size[0]}x{calib_size[1]})")
    print(f"method        : {method}")
    print(f"plane depth   : {Z_cm:.1f} cm")
    if "board_tilt_deg" in extra:
        print(f"board in frame: {100*extra['board_frame_frac']:.0f}% of width, "
              f"{extra['board_square_px']:.1f} px squares")
        print(f"board tilt    : {extra['board_tilt_deg']:.0f} deg off fronto-parallel")
    print(f"\ncm per pixel  : x {cm_per_px_x:.5f}   y {cm_per_px_y:.5f}")
    if "cm_per_px_direct" in extra:
        d = extra["cm_per_px_direct"]
        print(f"cross-check   : {d:.5f} cm/px from raw corner spacing "
              f"({100*abs(d - cm_per_px_x)/cm_per_px_x:.1f}% from x)")
    print(f"frame spans   : {w*cm_per_px_x:.0f} cm wide x {h*cm_per_px_y:.0f} cm tall "
          "at that depth -- does that match the room?")
    print(f"\n1 px of sway  = {10*cm_per_px_x:.2f} mm      "
          f"1 cm of sway = {1/cm_per_px_x:.1f} px")

    warn = []
    if 1.0 / cm_per_px_x < 3.0:
        warn.append(f"Only {1/cm_per_px_x:.1f} px per cm. Quiet-standing sway is a "
                    "couple of cm, so it would live in a handful of pixels. Move the "
                    "camera closer or record at a higher resolution.")
    if extra.get("board_tilt_deg", 0) > 15:
        warn.append(f"Board is {extra['board_tilt_deg']:.0f} deg off fronto-parallel. "
                    "Re-shoot it square to the camera so its plane is the subject's.")
    if "cm_per_px_direct" in extra and extra.get("board_tilt_deg", 0) <= 15:
        if abs(extra["cm_per_px_direct"] - cm_per_px_x) / cm_per_px_x > 0.05:
            warn.append("solvePnP and the raw corner spacing disagree by >5%. "
                        "Suspect the intrinsics, or a wrong --square-mm.")
    if rms > 0.5:
        warn.append(f"The intrinsics themselves have RMS {rms:.2f} px -- "
                    "fix the calibration before trusting this scale.")
    if warn:
        print("\nWARNINGS")
        for wmsg in warn:
            print(f"  ! {wmsg}")

    payload = {"session": args.session, "note": args.note,
               "measured_at": datetime.datetime.now().isoformat(timespec="seconds"),
               "method": method, "intrinsics": os.path.basename(args.intrinsics),
               "plane_depth_cm": Z_cm, "fx": fx, "fy": fy,
               "cm_per_px_x": cm_per_px_x, "cm_per_px_y": cm_per_px_y,
               "image_size": [w, h], "intrinsics_rms_px": rms, **extra}
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nsaved -> {args.out}")
    print("Multiply com_x_px / com_y_px in the .com.csv by cm_per_px_x / _y.")


if __name__ == "__main__":
    main()
