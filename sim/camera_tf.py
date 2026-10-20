#!/usr/bin/env python3
"""Print the static TF that matches the sim camera's pose in the world file.

The camera lives in the world (the robot's URDF has no camera), so nothing in
the robot description describes where it is. The behaviour tree transforms a
pose from the image frame to base_link, so that transform has to be published
explicitly and has to match the spawn pose, or the arm aims at the wrong point
(measured: 0.62 m instead of 0.34 m before this existed).
"""
from __future__ import annotations

import math
from pathlib import Path

# The camera is spawned by follow_camera.py, so the offset lives there; the world
# file no longer contains a robot camera at all.
# The camera model sits where the robot's camera is at the grasp station, so the
# offset from base_link is the station pose (0.76, 0, 0.10) minus base_link's
# world position at that moment (0.70, 0, 0.06).
CAMERA_OFFSET_XYZ = (0.155, 0.0, 0.18)
CAMERA_OFFSET_RPY = (0.0, 0.45, 0.0)
# The offset is already relative to base_link; nothing to subtract.


def rpy2mat(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr]]


def matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)]
            for i in range(3)]


def mat2quat(m):
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return [(m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s,
                (m[1][0] - m[0][1]) / s, 0.25 * s]
    if m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        return [0.25 * s, (m[0][1] + m[1][0]) / s,
                (m[0][2] + m[2][0]) / s, (m[2][1] - m[1][2]) / s]
    if m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        return [(m[0][1] + m[1][0]) / s, 0.25 * s,
                (m[1][2] + m[2][1]) / s, (m[0][2] - m[2][0]) / s]
    s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
    return [(m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s,
            0.25 * s, (m[1][0] - m[0][1]) / s]


def main():
    cam_xyz, cam_rpy = CAMERA_OFFSET_XYZ, CAMERA_OFFSET_RPY
    # model frame (+X forward, +Y left, +Z up) -> optical (+Z fwd, +X right, +Y down)
    r_opt = [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
    r_world_optical = matmul(rpy2mat(*cam_rpy), r_opt)
    delta = list(cam_xyz)
    quat = mat2quat(r_world_optical)
    print(' '.join(f'{value:.6f}' for value in (*delta, *quat)))


if __name__ == '__main__':
    main()
