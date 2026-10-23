#!/usr/bin/env python3
"""Pins the sim's shared point/orientation arithmetic.

These are the functions whose disagreement produced "the arm aimed at the wrong point"
bugs, so the properties are asserted directly: a known rotation, a round trip through the
matrix form, and the two spellings that used to coexist (rotate-then-translate against a
matrix multiply) agreeing.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pose_math  # noqa: E402


def test_identity_leaves_points_alone():
    assert pose_math.rotate((0.0, 0.0, 0.0, 1.0), (1.0, 2.0, 3.0)) == (1.0, 2.0, 3.0)


def test_quarter_turn_about_z():
    quaternion = pose_math.axis_angle((0.0, 0.0, 1.0), 90.0)
    x, y, z = pose_math.rotate(quaternion, (1.0, 0.0, 0.0))
    assert (round(x, 9), round(y, 9), round(z, 9)) == (0.0, 1.0, 0.0)


def test_translation_is_added_after_the_rotation():
    quaternion = pose_math.axis_angle((0.0, 0.0, 1.0), 90.0)
    assert [round(v, 9) for v in
            pose_math.transform_point(quaternion, (0.5, 0.0, 0.0), (1.0, 0.0, 0.0))] \
        == [0.5, 1.0, 0.0]


def test_rotation_matches_the_matrix_form():
    """The hand-written rotation and a matrix multiply must agree -- they coexisted."""
    quaternion = pose_math.multiply(pose_math.axis_angle((0.0, 1.0, 0.0), 30.0),
                                    pose_math.axis_angle((1.0, 0.0, 0.0), -20.0))
    matrix = pose_math.matrix_from_quaternion(quaternion)
    point = (0.31, -0.02, 0.05)
    by_quaternion = pose_math.rotate(quaternion, point)
    by_matrix = tuple(sum(matrix[row][column] * point[column] for column in range(3))
                      for row in range(3))
    assert all(abs(a - b) < 1e-12 for a, b in zip(by_quaternion, by_matrix))


def test_matrix_round_trip_returns_the_same_orientation():
    quaternion = pose_math.axis_angle((0.0, 0.0, 1.0), 123.0)
    back = pose_math.quaternion_from_matrix(pose_math.matrix_from_quaternion(quaternion))
    assert pose_math.angle_between(quaternion, back) < 1e-6


def test_angle_between_is_symmetric_and_zero_for_one_orientation():
    first = pose_math.axis_angle((0.0, 1.0, 0.0), 40.0)
    second = pose_math.axis_angle((0.0, 1.0, 0.0), 55.0)
    assert pose_math.angle_between(first, first) == pytest.approx(0.0, abs=1e-9)
    assert pose_math.angle_between(first, second) == pytest.approx(
        pose_math.angle_between(second, first), abs=1e-9)
    assert pose_math.angle_between(first, second) == pytest.approx(15.0, abs=1e-6)


def test_multiplication_order_applies_the_second_first():
    first = pose_math.axis_angle((0.0, 1.0, 0.0), 90.0)
    second = pose_math.axis_angle((0.0, 0.0, 1.0), 90.0)
    composed = pose_math.multiply(first, second)
    point = (1.0, 0.0, 0.0)
    stepwise = pose_math.rotate(first, pose_math.rotate(second, point))
    together = pose_math.rotate(composed, point)
    assert all(abs(a - b) < 1e-12 for a, b in zip(stepwise, together))
    # second turns +x into +y, first is about +y and leaves +y alone
    assert [round(v, 9) for v in stepwise] == [0.0, 1.0, 0.0]
