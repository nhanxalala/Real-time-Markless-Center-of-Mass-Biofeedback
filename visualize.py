"""Drawing helpers: skeleton, CoM marker, sway trail, and a stability target."""

from __future__ import annotations
import numpy as np
import cv2

from com import SEGMENTS, _point

_WHITE = (255, 255, 255)
_COM = (0, 0, 255)       # red (BGR)
_TRAIL = (0, 200, 255)   # amber
_OK = (0, 200, 0)        # green
_WARN = (0, 0, 255)      # red


def draw_skeleton(frame, kps_px, visibility=None, vis_thresh=0.5):
    for seg in SEGMENTS:
        p = _point(kps_px, seg.proximal).astype(int)
        d = _point(kps_px, seg.distal).astype(int)
        cv2.line(frame, tuple(p), tuple(d), _WHITE, 2, cv2.LINE_AA)
    return frame


def draw_com(frame, com_px, trail=None, target_px=None, target_radius_px=None):
    """Draw the CoM marker, optional sway trail, and optional stability target.

    Returns whether the CoM is inside the target (None if no target given).
    """
    inside = None
    if target_px is not None and target_radius_px is not None:
        dist = float(np.hypot(com_px[0] - target_px[0], com_px[1] - target_px[1]))
        inside = dist <= target_radius_px
        color = _OK if inside else _WARN
        cv2.circle(frame, tuple(np.asarray(target_px).astype(int)),
                   int(target_radius_px), color, 2, cv2.LINE_AA)

    if trail is not None and len(trail) > 1:
        pts = np.asarray(trail).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], False, _TRAIL, 2, cv2.LINE_AA)

    c = np.asarray(com_px).astype(int)
    cv2.circle(frame, tuple(c), 7, _COM, -1, cv2.LINE_AA)
    cv2.circle(frame, tuple(c), 9, _WHITE, 1, cv2.LINE_AA)
    return inside


def put_hud(frame, lines, org=(10, 24)):
    x, y = org
    for i, txt in enumerate(lines):
        cv2.putText(frame, txt, (x, y + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, txt, (x, y + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, _WHITE, 1, cv2.LINE_AA)
    return frame
