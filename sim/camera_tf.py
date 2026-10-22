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
followed by that frame change.

Output is the argument list for tf2_ros static_transform_publisher:
    x y z qx qy qz qw
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import camera_mount  # noqa: E402

#: Mount frame -> optical frame.
OPTICAL_FROM_MOUNT = ((0.0, 0.0, 1.0),
                      (-1.0, 0.0, 0.0),
                      (0.0, -1.0, 0.0))


def rpy_matrix(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return ((cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr))


def matmul(a, b):
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(3))
                       for j in range(3)) for i in range(3))


def matrix_quaternion(m):
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        return ((m[2][1] - m[1][2]) / scale, (m[0][2] - m[2][0]) / scale,
                (m[1][0] - m[0][1]) / scale, 0.25 * scale)
    if m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        scale = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        return (0.25 * scale, (m[0][1] + m[1][0]) / scale,
                (m[0][2] + m[2][0]) / scale, (m[2][1] - m[1][2]) / scale)
    if m[1][1] > m[2][2]:
        scale = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        return ((m[0][1] + m[1][0]) / scale, 0.25 * scale,
                (m[1][2] + m[2][1]) / scale, (m[0][2] - m[2][0]) / scale)
    scale = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
    return ((m[0][2] + m[2][0]) / scale, (m[1][2] + m[2][1]) / scale,
            0.25 * scale, (m[1][0] - m[0][1]) / scale)


def main():
    rotation = matmul(rpy_matrix(*camera_mount.OFFSET_RPY), OPTICAL_FROM_MOUNT)
    quaternion = matrix_quaternion(rotation)
    values = (*camera_mount.OFFSET_XYZ, *quaternion)
    print(' '.join('%.6f' % value for value in values))


if __name__ == '__main__':
    main()
