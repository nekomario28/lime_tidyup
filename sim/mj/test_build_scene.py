"""Regression tests for MuJoCo scene conventions that differ from ROS conventions."""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mj.build_scene as build_scene  # noqa: E402


def test_camera_quaternion_is_serialized_in_mjcf_wxyz_order():
    """Prevent the 57 mm vertical pose error caused by writing ROS xyzw as MJCF quat."""
    scene = build_scene.add_camera('<worldbody>\n    <body name="base_footprint">',
                                   (0.059, 0.0, 0.058), (0.0, 0.0, 0.0),
                                   resolution=(848, 480))
    match = re.search(r'<camera name="sim_camera"[^>]*quat="([^"]+)"', scene)
    assert match is not None
    values = tuple(float(value) for value in match.group(1).split())

    # This mount produces xyzw=(-0.5, 0.5, 0.5, -0.5). MJCF consumes wxyz.
    assert values == (-0.5, -0.5, 0.5, 0.5)


def test_cube_yaw_is_serialized_in_mjcf_wxyz_order():
    scene = build_scene.add_cube('  <worldbody>\n', 1.0, yaw_deg=60.0)
    match = re.search(r'<body name="cube"[^>]*quat="([^"]+)"', scene)
    assert match is not None
    values = tuple(float(value) for value in match.group(1).split())

    # MJCF uses wxyz: a 60-degree yaw has w=cos(30) and z=sin(30).
    assert values == (0.866025, 0.0, 0.0, 0.5)
