#!/usr/bin/env python3
"""Print the static transform base_link -> camera optical frame.

One fact, three places: the mount link in the description (so Gazebo puts the
sensor there), the frame name on the image (so a consumer knows which frame a pose
is in), and this transform (so a pose can be moved into base_link). All three come
from sim/camera_mount.py, because a mismatch between them is invisible until
something aims at the wrong point (measured: 0.62 m instead of 0.34 m when the
sensor and the transform disagreed).

The mount frame is the robot's (+x forward, +y left, +z up); a camera optical
frame is (+z forward, +x right, +y down), so the rotation is the mount's rpy
followed by that frame change. The arithmetic itself lives in sim/pose_math.py,
next to the rotation the rest of the sim uses.

Output is the argument list for tf2_ros static_transform_publisher:
    x y z qx qy qz qw
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import camera_mount  # noqa: E402
import pose_math  # noqa: E402

#: Mount frame -> optical frame.
OPTICAL_FROM_MOUNT = ((0.0, 0.0, 1.0),
                      (-1.0, 0.0, 0.0),
                      (0.0, -1.0, 0.0))


def rpy_matrix(roll, pitch, yaw):
    """The rotation matrix of a roll/pitch/yaw, as rows."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return ((cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr))


def matrix_product(first, second):
    return tuple(tuple(sum(first[row][k] * second[k][column] for k in range(3))
                       for column in range(3)) for row in range(3))


def main():
    rotation = matrix_product(rpy_matrix(*camera_mount.OFFSET_RPY),
                              OPTICAL_FROM_MOUNT)
    quaternion = pose_math.quaternion_from_matrix(rotation)
    values = (*camera_mount.OFFSET_XYZ, *quaternion)
    print(' '.join('%.6f' % value for value in values))


if __name__ == '__main__':
    main()
