# Markerless CoM pipeline — Phase 1 (offline)

Single-camera centre-of-mass estimation from recorded video: pose → BSP table →
whole-body CoM, overlaid on the clip and dumped to CSV. This is the offline core
(Phase 1 of the proposal). A live feasibility loop (`run_online.py`) is also
here now, to measure real-time latency; the multi-camera reference still comes
later.

## Layout

| file | role |
|---|---|
| `bsp_tables.py` | **Swappable** BSP tables. Dempster primary; de Leva (M/F) for sensitivity checks. Mass-sum validation. |
| `com.py` | Segment layout (landmark → segment) + whole-body CoM math. No MediaPipe dependency. |
| `pose_estimator.py` | MediaPipe Pose wrapper. Swap this to change pose model. |
| `visualize.py` | Skeleton, CoM marker, sway trail, stability target. |
| `run_offline.py` | CLI: video in → annotated video + `.com.csv`. |
| `run_online.py` | CLI: webcam (or file, for rehearsal) → live overlay window + real-time latency stats, optional `--record` to also dump `.annotated.mp4`/`.com.csv`. |
| `analyze.py` | Quick sway-trace / CoM-path plot from a CSV. |
| `test_com.py` | Pure-logic self-tests (no MediaPipe/video needed). |
| `calibrate_camera.py` | Checkerboard → intrinsics (fx, fy, cx, cy) + distortion. |
| `scale_from_board.py` | Intrinsics + plane depth (Z) → cm/px for the CoM signal. Z is **per session** — `--session` tags each measurement. |

## Setup (Windows, Python 3.10)

```bash
py -3.10 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

The example clips (`my_side_clip.mp4`, `my_front_clip.mp4`) are not included
in this repo; substitute your own recordings. The resulting `.com.csv` and
`.com.png` outputs from those clips are included for reference.

```bash
# self-tests first (no deps beyond numpy) — proves the CoM math
python test_com.py

# side clip, Dempster (primary — and the default, so the flag is optional)
python run_offline.py my_side_clip.mp4 --view side --table dempster

# front clip
python run_offline.py my_front_clip.mp4 --view front --table dempster

# swap the BSP table for the sensitivity check — one flag
python run_offline.py my_side_clip.mp4 --view side --table deleva --sex male

# demo the biofeedback element (stability target circle)
python run_offline.py my_side_clip.mp4 --view side --target

# plot the sway trace
python analyze.py my_side_clip.com.csv --view side
```

## Live loop (feasibility demo)

`run_online.py` runs the same pose → BSP → CoM → draw stages as
`run_offline.py`, but against a live camera (or a file, for rehearsing
without a webcam), with an on-screen HUD reporting real-time latency —
capture / pose / CoM+draw per stage, plus total ms and live fps, smoothed
over the last 30 frames. On quit (`q` or Esc) it prints mean/p95 latency per
stage and mean/median/p95 total latency — the number to cite for the paper's
real-time-latency claim, as opposed to `run_offline.py`'s "eff. fps", which
is offline processing throughput only.

```bash
# webcam 0, dempster table, no recording
python run_online.py

# rehearse against an existing clip instead of a webcam
python run_online.py --input my_side_clip.mp4

# also write live.annotated.mp4 / live.com.csv
python run_online.py --camera 0 --record --outdir results/

# same --table / --sex / --view / --target / --intrinsics / --scale flags as run_offline.py
python run_online.py --camera 0 --table deleva --sex male --view side --target
```

Live sources can misreport their frame size on the first read; if
`--intrinsics` doesn't match, `run_online.py` warns and skips undistort
rather than exiting (a file source still hard-errors, like
`run_offline.py`, since its resolution is reliably known upfront).

## What this does and does NOT claim

- **Sway is reported in image (2D) coordinates.** Front view → the horizontal axis
  is medio-lateral; side view → it's antero-posterior. This is the correct
  single-camera signal for Phase 1.
- **Metric units need the calibration stage below.** The `com_*_norm` /
  `com_*_px` columns are the raw signal. A single-camera scale is only valid for
  motion *in the calibrated plane* — which is the case here (front view → ML
  sway is in-plane; side view → AP sway is in-plane). True 3D ground-plane
  projection still needs the multi-camera reference.
- The printed "eff. fps" is **offline processing throughput**, not the live-loop
  end-to-end latency the paper will report.

## Calibration: pixels → centimetres

A checkerboard gives **intrinsics**, i.e. a ray per pixel. It does *not* give a
length — that needs the depth of the plane the motion happens in:

```
cm_per_px = Z_cm / fx
```

### Two different cadences

| | how often | why |
|---|---|---|
| **Intrinsics** (`calibrate_camera.py`) | **once** per camera + resolution | fx, fy, cx, cy and distortion are properties of the lens and sensor mode. They don't change between sessions. Re-solve only if you change camera, change resolution, or the lens gets knocked. |
| **Scale / Z** (`scale_from_board.py`) | **every session, and every view** | Z is the camera-to-subject plane depth. Move the tripod or stand the subject at a different distance and Z changed, so cm/px changed. Front and side views are two different planes → two scale files per session. |

Tag every scale measurement with `--session` so the file that produced a given
set of centimetres stays identifiable when you write the methods section. The
run scripts print the session (and note) of whatever `--scale` you hand them,
and warn if the scale was derived from a *different* intrinsics file than the
`--intrinsics` you passed — the realistic mistake is pairing this session's
scale with last session's intrinsics.

```bash
# 1. shoot 15-25 board views (SPACE keeps a shot). Autofocus OFF, and use the
#    SAME resolution you will record subjects at.
python calibrate_camera.py capture --cols 9 --rows 6 --square-mm 25.0 --width 1280 --height 720

# 2. fit intrinsics + distortion. Delete any view it flags as an outlier, re-solve.
python calibrate_camera.py solve --cols 9 --rows 6 --square-mm 25.0

# 3a. scale from a tape-measured subject distance (recommended at 2-3 m).
#     --session names the output scale_<session>.json and is stamped into it.
python scale_from_board.py --intrinsics camera_1280x720.npz --distance-cm 280 \
    --session s01_front --note "front view, tripod 2.8 m, subject barefoot"

# ...and the side view of the same session is a DIFFERENT plane depth
python scale_from_board.py --intrinsics camera_1280x720.npz --distance-cm 300 \
    --session s01_side --note "side view, tripod 3.0 m"

# 3b. or from a board held in the subject's plane at CoM height (needs the board
#     to span >=15% of the frame, else the tool refuses rather than guess)
python scale_from_board.py --intrinsics camera_1280x720.npz --board plane.png \
    --cols 9 --rows 6 --square-mm 60.0 --session s01_front

# 4. then feed that session's scale to either run script
python run_offline.py front_s01.mp4 --view front \
    --intrinsics camera_1280x720.npz --scale scale_s01_front.json
```

`--cols`/`--rows` are **inner corners**, not squares: a 10×7-square board has
9×6 inner corners. `--square-mm` must be ruler-measured off the printed board —
every centimetre this pipeline reports is scaled by that one number.

Both run scripts undistort when given `--intrinsics`: they remap the whole
frame (`cv2.initUndistortRectifyMap` + `cv2.remap`) so image, skeleton and CoM
marker all agree. Possible optimisation for the live loop: undistort only the
33 landmark *points* via `cv2.undistortPoints`, which is far cheaper than a
full-frame remap — the displayed frame doesn't need to be metric. Not currently
a bottleneck (pose inference dominates at ~30 ms/frame vs ~0.7 ms for
CoM+draw+remap combined), so it's unwired by choice, not oversight.

## Before you publish numbers

`bsp_tables.py` carries an accuracy caveat: the **head** and **trunk** coefficients
depend on your exact segment-endpoint definition. `validate_table()` only checks
that masses sum to 1.0 — it cannot verify a coefficient is the *right* one. Confirm
every coefficient against the original de Leva (1996) / Dempster (1955) tables for
your endpoint choice before reporting.
