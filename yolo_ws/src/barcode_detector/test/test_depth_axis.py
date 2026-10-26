"""The depth pose route, against a rendered cube and floor.

The renderer is a few ray intersections: it is the only way to know the true
orientation that the estimator is supposed to recover, and it makes the test
deterministic without a camera or a robot.
"""

import math

import numpy as np
import pytest

from barcode_detector.cube_depth_pose import (
    CubeDepthPoseError,
    _dense_band,
    _dominant_direction,
    _quaternion_from_axes,
    estimate_cube_depth_pose,
)

SIDE_M = 0.057
WIDTH, HEIGHT = 848, 480
CAMERA_MATRIX = (600.0, 0.0, 424.0, 0.0, 600.0, 240.0, 0.0, 0.0, 1.0)


def _look_at(position, target, up=(0.0, 0.0, 1.0)):
    """Camera-to-world rotation for a camera at ``position`` looking at target."""
    forward = np.array(target, dtype=float) - np.array(position, dtype=float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array(up, dtype=float))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.stack([right, down, forward], axis=1)


def _z_rotation(degrees):
    angle = math.radians(degrees)
    return np.array([
        [math.cos(angle), -math.sin(angle), 0.0],
        [math.sin(angle), math.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])


def _render(camera_position, cube_centre, yaw_deg, *, side=SIDE_M):
    """Depth in metres: the cube and the floor, nearest surface per pixel."""
    rotation = _look_at(camera_position, cube_centre)
    origin = np.array(camera_position, dtype=float)
    to_cube_frame = _z_rotation(-yaw_deg)
    box_origin = to_cube_frame @ (origin - np.array(cube_centre, dtype=float))
    k = np.asarray(CAMERA_MATRIX).reshape(3, 3)
    depth = np.full((HEIGHT, WIDTH), np.inf)
    on_cube = np.zeros((HEIGHT, WIDTH), dtype=bool)
    half = side / 2.0

    v, u = np.mgrid[0:HEIGHT, 0:WIDTH]
    direction = np.stack([
        (u - k[0, 2]) / k[0, 0],
        (v - k[1, 2]) / k[1, 1],
        np.ones_like(u),
    ], axis=-1)
    world_direction = direction @ rotation.T

    # Floor at z = 0, seen from above.
    down_axis = world_direction[..., 2]
    floor = np.where(down_axis < -1e-9, -origin[2] / down_axis, np.inf)

    # Cube: slab intersection in the cube's own frame.
    box_direction = world_direction @ to_cube_frame.T
    near = np.full((HEIGHT, WIDTH), -np.inf)
    far = np.full((HEIGHT, WIDTH), np.inf)
    hit = np.ones((HEIGHT, WIDTH), dtype=bool)
    for axis in range(3):
        lo = (-half - box_origin[axis]) / box_direction[..., axis]
        hi = (half - box_origin[axis]) / box_direction[..., axis]
        near = np.maximum(near, np.minimum(lo, hi))
        far = np.minimum(far, np.maximum(lo, hi))
        hit &= far >= np.maximum(near, 0.0)
    cube = np.where(hit & (near > 0.0), near, np.inf)

    depth = np.minimum(floor, cube)
    on_cube = np.isfinite(cube) & (cube <= floor)
    depth[~np.isfinite(depth)] = np.nan
    return depth, on_cube, rotation


def _box_of(mask, shrink=0.0):
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        raise AssertionError('the renderer drew no cube')
    x0, x1 = cols.min(), cols.max() + 1
    y0, y1 = rows.min(), rows.max() + 1
    inset_x = shrink * (x1 - x0)
    inset_y = shrink * (y1 - y0)
    return [x0 + inset_x, y0 + inset_y, x1 - inset_x, y1 - inset_y]


def _quarter_turn_difference(measured, expected):
    """Difference between two angles once a quarter turn is taken as equal."""
    return ((measured - expected + 45.0) % 90.0) - 45.0


@pytest.mark.parametrize('yaw_deg', [0.0, 20.0, 45.0])
def test_recovers_the_cube_frame_from_depth(yaw_deg):
    cube_centre = (0.0, 0.0, SIDE_M / 2.0 + 0.0005)
    camera_position = (0.0, -0.30, 0.23)
    depth, on_cube, rotation = _render(camera_position, cube_centre, yaw_deg)

    pose = estimate_cube_depth_pose(
        depth,
        _box_of(on_cube, shrink=0.05),
        CAMERA_MATRIX,
        side_length_m=SIDE_M,
    )

    true_centre_camera = rotation.T @ (
        np.array(cube_centre) - np.array(camera_position))
    assert np.allclose(pose.position_m, true_centre_camera, atol=0.005)

    axes = np.array(pose.orientation_xyzw)
    rotation_cube = np.stack([
        _rotate(axes, [1.0, 0.0, 0.0]),
        _rotate(axes, [0.0, 1.0, 0.0]),
        _rotate(axes, [0.0, 0.0, 1.0]),
    ], axis=1)
    horizontal_world = rotation @ rotation_cube[:, 0]
    measured_yaw = math.degrees(
        math.atan2(horizontal_world[1], horizontal_world[0]))
    assert abs(_quarter_turn_difference(measured_yaw, yaw_deg)) < 3.0

    assert abs(np.linalg.det(rotation_cube) - 1.0) < 1e-9
    assert np.allclose(rotation_cube[:, 2], pose.up_xyz, atol=1e-9)

    up_world = rotation @ np.array(pose.up_xyz)
    assert abs(math.degrees(math.acos(
        float(np.clip(up_world @ np.array([0.0, 0.0, 1.0]), -1.0, 1.0))))) < 3.0


def test_up_hint_needs_no_floor_at_all(yaw_deg=30.0):
    """With the up axis supplied, a box with no floor below it still solves."""
    cube_centre = (0.0, 0.0, SIDE_M / 2.0 + 0.0005)
    camera_position = (0.0, -0.30, 0.23)
    depth, on_cube, rotation = _render(camera_position, cube_centre, yaw_deg)
    box = _box_of(on_cube)
    # Drop everything the depth pose could fit a floor from: the whole region
    # below the box, which is where it looks for one.
    depth[int(box[3]) + 1:, :] = np.nan

    with pytest.raises(CubeDepthPoseError):
        estimate_cube_depth_pose(
            depth, box, CAMERA_MATRIX, side_length_m=SIDE_M)

    up_hint = rotation.T @ np.array([0.0, 0.0, 1.0])
    pose = estimate_cube_depth_pose(
        depth, box, CAMERA_MATRIX, side_length_m=SIDE_M, up_hint_xyz=up_hint)

    assert pose.up_source == 'up_hint'
    true_centre_camera = rotation.T @ (
        np.array(cube_centre) - np.array(camera_position))
    assert np.allclose(pose.position_m, true_centre_camera, atol=0.005)
    axes = np.array(pose.orientation_xyzw)
    horizontal = _rotate(axes, [1.0, 0.0, 0.0])
    world = rotation @ horizontal
    measured_yaw = math.degrees(math.atan2(world[1], world[0]))
    assert abs(_quarter_turn_difference(measured_yaw, yaw_deg)) < 3.0


def test_withholds_when_no_vertical_face_is_visible():
    cube_centre = (0.0, 0.0, SIDE_M / 2.0)
    camera_position = (0.0, -0.02, 0.40)
    depth, on_cube, _ = _render(camera_position, cube_centre, 0.0)

    with pytest.raises(CubeDepthPoseError):
        estimate_cube_depth_pose(
            depth,
            _box_of(on_cube, shrink=0.05),
            CAMERA_MATRIX,
            side_length_m=SIDE_M,
        )


def test_withholds_when_the_depth_is_empty():
    depth = np.full((HEIGHT, WIDTH), np.nan)

    with pytest.raises(CubeDepthPoseError):
        estimate_cube_depth_pose(
            depth, [380.0, 250.0, 470.0, 340.0], CAMERA_MATRIX)


def test_quaternion_from_axes_round_trips():
    quaternion = _quaternion_from_axes(
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    )
    assert quaternion == (0.0, 0.0, 0.0, 1.0)


def _rotate(quaternion_xyzw, vector):
    """Rotate a vector by an XYZW quaternion."""
    x, y, z, w = (float(value) for value in quaternion_xyzw)
    matrix = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return matrix @ np.array(vector, dtype=float)

def test_dominant_direction_survives_mixed_normal_signs():
    """A plane's pixel normals can disagree about the sign; the mode must not.

    Measured on the robot 2026-10-26: averaging a sign-folded cluster without
    aligning it first cancelled to a direction no normal agreed with, which sent
    the axis 45 degrees off the cube.
    """
    direction = np.array([-0.484, 0.484, 0.729])
    normals = np.tile(direction, (40, 1))
    normals[1::2] = -normals[1::2]
    normals += np.linspace(-0.001, 0.001, num=normals.shape[0])[:, None]

    measured, count, total = _dominant_direction(normals)

    assert total == normals.shape[0]
    assert count == normals.shape[0]
    assert abs(float(measured @ direction)) > 0.999

def test_dense_band_prefers_the_populated_plane_over_a_sparse_tail():
    """A spread face plane must win against a few points further along.

    Measured on the robot 2026-10-26: a face spread 6 mm along the axis, and the
    furthest occupied bin held 243 of its 3000 points; choosing that bin left the
    face with a quarter of its pixels and the approach was refused.
    """
    spread_face = np.linspace(-0.0100, -0.0040, num=3000)
    sparse_tail = np.linspace(-0.0028, -0.0024, num=240)
    values = np.concatenate([spread_face, sparse_tail])

    measured = _dense_band(values)

    assert abs(measured - float(np.mean(spread_face))) < 0.0005
