"""Validation helpers for optional cube 3D-pose evidence."""

from __future__ import annotations

import math
import time


def accepted_pose_distance(
    detection,
    *,
    maximum_reprojection_error_px=4.0,
    maximum_age_sec=3.0,
    minimum_inliers=4,
    now=None
):
    """
    Return camera-forward distance when optional pose evidence is valid.

    ``None`` means the caller must use its existing depth fallback.  This
    function never turns perception evidence into motion authority by itself.
    """
    if not isinstance(detection, dict):
        return None

    pose = detection.get('pose_3d')
    if not isinstance(pose, dict):
        return None
    if pose.get('geometry_id') != 'cube-axis-v1':
        return None
    if not pose.get('frame_id'):
        return None

    try:
        position = pose['position_m']
        distance = float(position[2])
        reprojection_error = float(pose['reprojection_error_px'])
        correspondence_count = int(pose['correspondence_count'])
        inlier_count = int(pose['inlier_count'])
        timestamp = float(pose['timestamp'])
    except (KeyError, IndexError, TypeError, ValueError):
        return None

    values = (distance, reprojection_error, timestamp)
    if not all(math.isfinite(value) for value in values):
        return None
    if not 0.05 <= distance <= 10.0:
        return None
    if not 0.0 <= reprojection_error <= float(
        maximum_reprojection_error_px
    ):
        return None
    if correspondence_count < 4 or inlier_count < int(minimum_inliers):
        return None

    if now is None:
        now = time.time()
    age = float(now) - timestamp
    if age < -1.0 or age > float(maximum_age_sec):
        return None

    return distance
