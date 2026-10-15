"""
Checkerboard camera calibration (Phase 2a: single-camera intrinsics).

Recovers the pinhole intrinsics (fx, fy, cx, cy) and lens distortion for ONE
camera at ONE resolution. Intrinsics alone do NOT give you centimetres -- they
give you a ray per pixel. Run `scale_from_board.py` afterwards to turn that
into cm/px at the subject's distance.

Usage:
    # 1a. grab board shots live from the webcam  (SPACE = keep, ESC = done)
    python calibrate_camera.py capture --cols 9 --rows 6 --square-mm 25.0

    # 1b. ...or pull them out of a clip you filmed of the board
    python calibrate_camera.py from-video board.mp4 --cols 9 --rows 6 --square-mm 25.0

    # 2.  solve
    python calibrate_camera.py solve --cols 9 --rows 6 --square-mm 25.0

--cols/--rows are INNER CORNERS, not squares: a board with 10x7 squares has
9x6 inner corners. Get this wrong and nothing is ever detected.

--square-mm must be measured with a ruler on the printed board. Every
centimetre this pipeline ever reports is scaled by that one number.
"""

from __future__ import annotations
import argparse
import glob
import json
import os

import cv2
import numpy as np

CORNER_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
SUBPIX_CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)


def object_points(cols: int, rows: int, square_mm: float) -> np.ndarray:
    """(cols*rows, 3) board-frame corner coordinates in mm, z=0."""
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    return objp * square_mm


def find_corners(gray: np.ndarray, pattern: tuple[int, int]):
    ok, corners = cv2.findChessboardCorners(gray, pattern, flags=CORNER_FLAGS)
    if not ok:
        return None
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), SUBPIX_CRIT)


# --- shot collection -------------------------------------------------------

def cmd_capture(args):
    os.makedirs(args.images, exist_ok=True)
    pattern = (args.cols, args.rows)
    cap = cv2.VideoCapture(args.device, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera device {args.device}")
    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    # Autofocus changes the focal length between shots and invalidates the whole
    # calibration. Many webcams ignore this request; report what actually stuck.
    af_off = cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Capturing at {w}x{h}  (autofocus-off accepted: {af_off})")
    print("!! This resolution MUST match the resolution you record subjects at.")
    print("SPACE = keep shot, ESC = done.\n")

    kept = len(glob.glob(os.path.join(args.images, "*.png")))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = find_corners(gray, pattern)
        view = frame.copy()
        if corners is not None:
            cv2.drawChessboardCorners(view, pattern, corners, True)
        msg = f"kept {kept}/{args.want}   board: {'FOUND' if corners is not None else '--'}"
        cv2.putText(view, msg, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if corners is not None else (0, 0, 255), 2)
        cv2.imshow("calibration capture", view)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        if k == 32 and corners is not None:
            path = os.path.join(args.images, f"calib_{kept:03d}.png")
            cv2.imwrite(path, frame)
            kept += 1
            print(f"  saved {path}")
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n{kept} shots in {args.images}/ -- now run `solve`.")


def cmd_from_video(args):
    os.makedirs(args.images, exist_ok=True)
    pattern = (args.cols, args.rows)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video is {w}x{h} -- must match your subject-clip resolution.")
    kept, idx = 0, 0
    while kept < args.want:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % args.every == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if find_corners(gray, pattern) is not None:
                path = os.path.join(args.images, f"calib_{kept:03d}.png")
                cv2.imwrite(path, frame)
                kept += 1
                print(f"  frame {idx} -> {path}")
        idx += 1
    cap.release()
    print(f"\n{kept} shots in {args.images}/ -- now run `solve`.")


# --- solve -----------------------------------------------------------------

def cmd_solve(args):
    pattern = (args.cols, args.rows)
    paths = sorted(glob.glob(os.path.join(args.images, "*.png")) +
                   glob.glob(os.path.join(args.images, "*.jpg")))
    if not paths:
        raise SystemExit(f"No images in {args.images}/")

    objp = object_points(args.cols, args.rows, args.square_mm)
    obj_pts, img_pts, used, size = [], [], [], None
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hw = gray.shape[::-1]
        if size is None:
            size = hw
        elif hw != size:
            raise SystemExit(
                f"{p} is {hw} but earlier images are {size}. Every calibration "
                "image must come from the same camera at the same resolution."
            )
        corners = find_corners(gray, pattern)
        if corners is None:
            print(f"  no board  : {os.path.basename(p)}")
            continue
        obj_pts.append(objp)
        img_pts.append(corners)
        used.append(p)

    print(f"\nUsable views: {len(used)}/{len(paths)}")
    if len(used) < 8:
        raise SystemExit("Need >= 8 usable views (15-25 is the sweet spot).")

    # A 5-coefficient distortion model overfits badly on a modest webcam set --
    # it will happily return k3 = -35 for a lens with no distortion at all. Fit
    # only as many terms as the data supports.
    flags = {
        "k1": cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST,
        "k1k2": cv2.CALIB_FIX_K3,
        "full": 0,
    }[args.dist_model]
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_pts, img_pts, size, None, None, flags=flags)

    # Per-view reprojection error -- one bad shot can drag fx off by percent.
    # cv2.norm gives sqrt(sum of squared residuals) over all coordinates, so
    # divide by sqrt(N) for a per-point RMS comparable to the overall figure.
    errs = []
    for i in range(len(used)):
        proj, _ = cv2.projectPoints(obj_pts[i], rvecs[i], tvecs[i], K, dist)
        n = len(proj)
        errs.append(float(cv2.norm(img_pts[i], proj, cv2.NORM_L2) / np.sqrt(n)))
    errs = np.asarray(errs)

    # Pose diversity: if every board is fronto-parallel, focal length and board
    # distance are near-degenerate and fx comes out confidently wrong.
    tilts = []
    for rv in rvecs:
        R, _ = cv2.Rodrigues(rv)
        tilts.append(np.degrees(np.arccos(np.clip(abs(R[2, 2]), -1.0, 1.0))))
    tilts = np.asarray(tilts)

    # Frame coverage: corners must reach the edges, where the distortion lives.
    all_c = np.concatenate(img_pts).reshape(-1, 2)
    cover = ((np.ptp(all_c[:, 0]) / size[0]) * (np.ptp(all_c[:, 1]) / size[1]))

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    print(f"\nimage size    : {size[0]}x{size[1]}")
    print(f"dist model    : {args.dist_model}")
    print(f"fx, fy        : {fx:.2f}, {fy:.2f} px   (aspect {fy/fx:.4f}, want ~1.00)")
    print(f"cx, cy        : {cx:.2f}, {cy:.2f} px   (frame centre {size[0]/2:.1f}, {size[1]/2:.1f})")
    print(f"distortion    : k1={dist[0][0]:+.4f} k2={dist[0][1]:+.4f} "
          f"p1={dist[0][2]:+.4f} p2={dist[0][3]:+.4f} k3={dist[0][4]:+.4f}")
    print(f"RMS reproj    : {rms:.4f} px   (want < 0.5; report this in the paper)")
    print(f"per-view err  : mean {errs.mean():.3f}  worst {errs.max():.3f} "
          f"({os.path.basename(used[int(errs.argmax())])})")
    print(f"board tilt    : max {tilts.max():.0f} deg, median {np.median(tilts):.0f} deg")
    print(f"frame coverage: {100*cover:.0f}% of the frame area")

    warn = []
    if rms > 0.5:
        warn.append("RMS > 0.5 px -- reshoot; usually one blurry or edge-clipped view.")
    if errs.max() > 3 * max(errs.mean(), 1e-6):
        warn.append(f"'{os.path.basename(used[int(errs.argmax())])}' is an outlier -- "
                    "delete it and re-solve.")
    if tilts.max() < 20:
        warn.append("All boards are nearly flat-on to the camera. Tilt it 30-45 deg "
                    "in several shots or fx is unreliable.")
    if cover < 0.6:
        warn.append("Corners never reach the frame edges -- the distortion "
                    "coefficients are extrapolated out there, which is exactly "
                    "where the feet and head sit.")
    if warn:
        print("\nWARNINGS")
        for w in warn:
            print(f"  ! {w}")

    out = args.out or f"camera_{size[0]}x{size[1]}.npz"
    np.savez(out, K=K, dist=dist, image_size=np.asarray(size),
             rms=rms, square_mm=args.square_mm,
             pattern=np.asarray(pattern), n_views=len(used),
             dist_model=args.dist_model)
    print(f"\nsaved -> {out}")
    json_path = os.path.splitext(out)[0] + ".json"
    with open(json_path, "w") as f:
        json.dump({"fx": fx, "fy": fy, "cx": cx, "cy": cy,
                   "dist": dist.ravel().tolist(),
                   "image_size": list(size), "rms_px": float(rms),
                   "square_mm": args.square_mm, "n_views": len(used),
                   "dist_model": args.dist_model}, f, indent=2)
    print(f"saved -> {json_path}  (paste into the methods section)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def board_args(p):
        p.add_argument("--cols", type=int, required=True, help="INNER corners across")
        p.add_argument("--rows", type=int, required=True, help="INNER corners down")
        p.add_argument("--square-mm", type=float, required=True,
                       help="measured square side, in mm")
        p.add_argument("--images", default="calib", help="image folder")

    c = sub.add_parser("capture", help="grab board shots from a live webcam")
    board_args(c)
    c.add_argument("--device", type=int, default=0)
    c.add_argument("--width", type=int, default=None)
    c.add_argument("--height", type=int, default=None)
    c.add_argument("--want", type=int, default=20)
    c.set_defaults(func=cmd_capture)

    v = sub.add_parser("from-video", help="pull board shots out of a clip")
    v.add_argument("video")
    board_args(v)
    v.add_argument("--every", type=int, default=15, help="sample every Nth frame")
    v.add_argument("--want", type=int, default=20)
    v.set_defaults(func=cmd_from_video)

    s = sub.add_parser("solve", help="fit intrinsics + distortion")
    board_args(s)
    s.add_argument("--out", default=None, help="output .npz (default camera_WxH.npz)")
    s.add_argument("--dist-model", default="k1k2", choices=["k1", "k1k2", "full"],
                   help="how many distortion terms to fit. k1k2 (default) is the "
                        "safe choice for a webcam; 'full' (k1,k2,k3,p1,p2) needs "
                        "20+ well-spread views or it overfits wildly.")
    s.set_defaults(func=cmd_solve)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
