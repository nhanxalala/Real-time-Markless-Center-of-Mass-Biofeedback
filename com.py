"""
Whole-body centre-of-mass (CoM) from 2D pose keypoints + a BSP table.

The segment LAYOUT (which landmarks bound each segment) is fixed here and is
independent of the BSP table, so you can swap de Leva <-> Dempster freely.

Landmark indices follow MediaPipe Pose / BlazePose (33 landmarks). If you use
a different pose model, remap in POSE_LANDMARKS / SEGMENTS only.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from bsp_tables import SegmentParam

# --- MediaPipe Pose landmark indices we use ---------------------------------
L = dict(
    nose=0, left_ear=7, right_ear=8,
    left_shoulder=11, right_shoulder=12,
    left_elbow=13, right_elbow=14,
    left_wrist=15, right_wrist=16,
    left_index=19, right_index=20,
    left_hip=23, right_hip=24,
    left_knee=25, right_knee=26,
    left_ankle=27, right_ankle=28,
    left_foot_index=31, right_foot_index=32,
)


@dataclass(frozen=True)
class Segment:
    """A segment instance: which type's BSP params to use, and its endpoints.

    An endpoint is either a single landmark index or a tuple of indices whose
    mean is taken (used for 'virtual' joints like mid-shoulder / mid-hip / head).
    """
    name: str
    seg_type: str
    proximal: int | tuple[int, ...]
    distal: int | tuple[int, ...]


# Head is approximated as ear-midpoint (proximal) -> neck/mid-shoulder (distal).
# Trunk is mid-shoulder (proximal) -> mid-hip (distal).
# These 'virtual joints' are the standard markerless approximation; see the
# caveat in bsp_tables.py.
_HEAD_TOP = (L["left_ear"], L["right_ear"])
_NECK = (L["left_shoulder"], L["right_shoulder"])
_PELVIS = (L["left_hip"], L["right_hip"])

SEGMENTS: tuple[Segment, ...] = (
    Segment("head",        "head",      _HEAD_TOP, _NECK),
    Segment("trunk",       "trunk",     _NECK, _PELVIS),
    Segment("upper_arm_L", "upper_arm", L["left_shoulder"],  L["left_elbow"]),
    Segment("upper_arm_R", "upper_arm", L["right_shoulder"], L["right_elbow"]),
    Segment("forearm_L",   "forearm",   L["left_elbow"],  L["left_wrist"]),
    Segment("forearm_R",   "forearm",   L["right_elbow"], L["right_wrist"]),
    Segment("hand_L",      "hand",      L["left_wrist"],  L["left_index"]),
    Segment("hand_R",      "hand",      L["right_wrist"], L["right_index"]),
    Segment("thigh_L",     "thigh",     L["left_hip"],  L["left_knee"]),
    Segment("thigh_R",     "thigh",     L["right_hip"], L["right_knee"]),
    Segment("shank_L",     "shank",     L["left_knee"],  L["left_ankle"]),
    Segment("shank_R",     "shank",     L["right_knee"], L["right_ankle"]),
    Segment("foot_L",      "foot",      L["left_ankle"],  L["left_foot_index"]),
    Segment("foot_R",      "foot",      L["right_ankle"], L["right_foot_index"]),
)


def _point(kps: np.ndarray, ref: int | tuple[int, ...]) -> np.ndarray:
    """Resolve an endpoint (single index or mean of several) to an (x, y)."""
    if isinstance(ref, tuple):
        return kps[list(ref)].mean(axis=0)
    return kps[ref]


def compute_com(
    kps: np.ndarray,
    table: dict[str, SegmentParam],
    visibility: np.ndarray | None = None,
    vis_thresh: float = 0.5,
) -> tuple[np.ndarray | None, dict]:
    """Whole-body CoM for one frame.

    kps        : (33, 2) array of landmark (x, y) in ANY consistent unit
                 (normalized 0..1 or pixels — the CoM comes out in the same unit).
    table      : a BSP table from bsp_tables.get_table(...)
    visibility : optional (33,) landmark confidence; segments whose endpoints
                 fall below vis_thresh are dropped and their mass redistributed.

    Returns (com_xy, info). com_xy is None if too little of the body is visible.
    info carries per-segment CoMs and the fraction of body mass actually used.
    """
    seg_coms = []
    seg_masses = []
    used = []
    for seg in SEGMENTS:
        if visibility is not None:
            idx = list(seg.proximal if isinstance(seg.proximal, tuple) else (seg.proximal,))
            idx += list(seg.distal if isinstance(seg.distal, tuple) else (seg.distal,))
            if np.min(visibility[idx]) < vis_thresh:
                continue
        p = _point(kps, seg.proximal)
        d = _point(kps, seg.distal)
        params = table[seg.seg_type]
        com = p + params.com * (d - p)
        seg_coms.append(com)
        seg_masses.append(params.mass)
        used.append(seg.name)

    mass_used = float(np.sum(seg_masses))
    if mass_used < 0.5:  # less than half the body visible -> untrustworthy
        return None, {"mass_used": mass_used, "segments_used": used}

    seg_coms = np.asarray(seg_coms)
    seg_masses = np.asarray(seg_masses)
    # Normalize by mass actually used (redistributes dropped-segment mass).
    com_xy = (seg_masses[:, None] * seg_coms).sum(axis=0) / mass_used
    return com_xy, {
        "mass_used": mass_used,
        "segments_used": used,
        "segment_coms": seg_coms,
    }
