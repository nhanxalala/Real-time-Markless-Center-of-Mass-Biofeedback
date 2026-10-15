"""
Thin wrapper around MediaPipe Pose. Isolated here so the CoM math (com.py) has
no hard dependency on MediaPipe and stays unit-testable.

Swap this file to use OpenPose / another model as long as you return the same
(33, 2) keypoint convention used in com.py's landmark map.
"""

from __future__ import annotations
import numpy as np

try:
    import mediapipe as mp
except ImportError as e:  # pragma: no cover - import guard
    raise ImportError(
        "mediapipe is not installed. Activate the venv and run "
        "`pip install -r requirements.txt` (use Python 3.10-3.12)."
    ) from e


class PoseEstimator:
    def __init__(self, model_complexity: int = 1, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def __call__(self, frame_bgr: np.ndarray):
        """Return (kps_norm, kps_px, visibility) or (None, None, None).

        kps_norm    : (33, 2) normalized 0..1 image coords (x right, y down)
        kps_px      : (33, 2) pixel coords
        visibility  : (33,) landmark confidence 0..1
        """
        import cv2
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self._pose.process(rgb)
        if not res.pose_landmarks:
            return None, None, None
        lm = res.pose_landmarks.landmark
        kps_norm = np.array([[p.x, p.y] for p in lm], dtype=np.float64)
        vis = np.array([p.visibility for p in lm], dtype=np.float64)
        kps_px = kps_norm * np.array([w, h])
        return kps_norm, kps_px, vis

    def close(self):
        self._pose.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
