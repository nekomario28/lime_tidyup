"""The tool yaw must come from the cube's own faces, in base_link."""

import math

from cm1.lib.actor.manipulator import face_aligned_yaw_deg


def _quaternion_from_rotation(columns):
    """XYZW quaternion of the rotation whose columns are the given axes."""
    matrix = [[columns[col][row] for col in range(3)] for row in range(3)]
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return (
            (matrix[2][1] - matrix[1][2]) / s,
            (matrix[0][2] - matrix[2][0]) / s,
            (matrix[1][0] - matrix[0][1]) / s,
            0.25 * s,
        )
    index = max(range(3), key=lambda i: matrix[i][i])
    if index == 0:
        s = math.sqrt(1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2]) * 2.0
        return (0.25 * s, (matrix[0][1] + matrix[1][0]) / s,
                (matrix[0][2] + matrix[2][0]) / s,
                (matrix[2][1] - matrix[1][2]) / s)
    if index == 1:
        s = math.sqrt(1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2]) * 2.0
        return ((matrix[0][1] + matrix[1][0]) / s, 0.25 * s,
                (matrix[1][2] + matrix[2][1]) / s,
                (matrix[0][2] - matrix[2][0]) / s)
    s = math.sqrt(1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1]) * 2.0
    return ((matrix[0][2] + matrix[2][0]) / s,
            (matrix[1][2] + matrix[2][1]) / s, 0.25 * s,
            (matrix[1][0] - matrix[0][1]) / s)


def _horizontal(yaw_deg):
    yaw = math.radians(yaw_deg)
    return (math.cos(yaw), math.sin(yaw), 0.0)


def _cube_standing(yaw_deg):
    """Cube frame in base_link: upright, turned by ``yaw`` about the vertical."""
    return _quaternion_from_rotation((
        _horizontal(yaw_deg),
        _horizontal(yaw_deg + 90.0),
        (0.0, 0.0, 1.0),
    ))


def _cube_rolled_onto_its_first_axis(yaw_deg):
    """Cube frame whose *first* axis points at the ceiling.

    Still a proper rotation, and the two sideways axes sit at ``yaw``: a consumer
    that always aligns the jaws with the first axis would aim them at the sky.
    """
    return _quaternion_from_rotation((
        (0.0, 0.0, 1.0),
        _horizontal(yaw_deg),
        _horizontal(yaw_deg + 90.0),
    ))


def _quarter_turn_difference(measured, expected):
    return ((measured - expected + 45.0) % 90.0) - 45.0


def test_yaw_follows_the_cube():
    for yaw_deg in (0.0, 30.0, 44.0):
        measured = face_aligned_yaw_deg(_cube_standing(yaw_deg))
        assert abs(_quarter_turn_difference(measured, yaw_deg)) < 1e-6


def test_yaw_is_snapped_to_the_nearest_quarter_turn():
    measured = face_aligned_yaw_deg(_cube_standing(80.0))
    assert abs(_quarter_turn_difference(measured, 80.0)) < 1e-6
    assert -45.0 <= measured < 45.0


def test_yaw_uses_a_sideways_cube_axis_whatever_axis_points_up():
    measured = face_aligned_yaw_deg(_cube_rolled_onto_its_first_axis(25.0))
    assert abs(_quarter_turn_difference(measured, 25.0)) < 1e-6
