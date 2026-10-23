#!/usr/bin/env python3
"""Point and orientation arithmetic shared by the sim tools.

One owner for the rotation every tool needs. The quaternion rotation of a point was
written out by hand in three places -- twice inside sim/grasp_cube.py (moving a cube
pose into base_link, and projecting the aim back into the image) and once, spelled
differently, in the recognition eval -- so a fix to the formula in one place would have
silently disagreed with the other two. It is the arithmetic behind "the arm aimed at the
wrong point" bugs, where the only symptom is a distance that is 5-60 cm off.

Dependency-free on purpose: unlike the perception package these are imported both by the
grasp script and by the eval tools, and only the latter have barcode_detector on their
path. Quaternions are xyzw tuples throughout, matching the message field they come from.
"""
from __future__ import annotations

import math


def rotate(quaternion_xyzw, point):
    """Rotate a point by a quaternion (the two-way product without the translation)."""
    x, y, z, w = quaternion_xyzw
    px, py, pz = point
    cross = (y * pz - z * py, z * px - x * pz, x * py - y * px)
    dot = x * px + y * py + z * pz
    norm = x * x + y * y + z * z
    return tuple(2.0 * dot * axis + (w * w - norm) * coordinate + 2.0 * w * component
                 for axis, coordinate, component
                 in zip((x, y, z), (px, py, pz), cross))


def transform_point(quaternion_xyzw, translation, point):
    """A point rotated and translated by one transform: p' = R p + t."""
    rotated = rotate(quaternion_xyzw, point)
    return tuple(rotated[i] + translation[i] for i in range(3))


def matrix_from_quaternion(quaternion_xyzw):
    """The 3x3 rotation matrix of a quaternion, as rows."""
    x, y, z, w = quaternion_xyzw
    return ((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)))


def quaternion_from_matrix(matrix):
    """The xyzw quaternion of a rotation matrix, by the largest-diagonal branch."""
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        return ((matrix[2][1] - matrix[1][2]) / scale,
                (matrix[0][2] - matrix[2][0]) / scale,
                (matrix[1][0] - matrix[0][1]) / scale, 0.25 * scale)
    if matrix[0][0] > matrix[1][1] and matrix[0][0] > matrix[2][2]:
        scale = math.sqrt(1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2]) * 2.0
        return (0.25 * scale, (matrix[0][1] + matrix[1][0]) / scale,
                (matrix[0][2] + matrix[2][0]) / scale,
                (matrix[2][1] - matrix[1][2]) / scale)
    if matrix[1][1] > matrix[2][2]:
        scale = math.sqrt(1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2]) * 2.0
        return ((matrix[0][1] + matrix[1][0]) / scale, 0.25 * scale,
                (matrix[1][2] + matrix[2][1]) / scale,
                (matrix[0][2] - matrix[2][0]) / scale)
    scale = math.sqrt(1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1]) * 2.0
    return ((matrix[0][2] + matrix[2][0]) / scale,
            (matrix[1][2] + matrix[2][1]) / scale, 0.25 * scale,
            (matrix[1][0] - matrix[0][1]) / scale)


def multiply(first, second):
    """The quaternion product, i.e. applying second then first."""
    ax, ay, az, aw = first
    bx, by, bz, bw = second
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def axis_angle(axis, degrees):
    """A rotation of `degrees` about an axis, as a quaternion."""
    length = math.sqrt(sum(component * component for component in axis))
    half = math.radians(degrees) / 2.0
    unit = [component / length for component in axis]
    return tuple(unit[i] * math.sin(half) for i in range(3)) + (math.cos(half),)


def angle_between(first, second):
    """The angle between two orientations, in degrees."""
    dot = abs(sum(a * b for a, b in zip(first, second)))
    return math.degrees(2.0 * math.acos(min(1.0, dot)))
