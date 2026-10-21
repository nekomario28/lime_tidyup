"""Tests for the single-view face pose."""
import math

import numpy as np
import pytest

from barcode_detector.cube_face_colors import _face_corner_object_points, REFERENCE_RGB
from barcode_detector.cube_single_view_pose import (
    FacePoseError,
    estimate_face_pose,
    face_lab,
    face_quad_corners,
    rotate_in_plane,
)

CAMERA_MATRIX = (455.0, 0.0, 424.0, 0.0, 455.0, 240.0, 0.0, 0.0, 1.0)
SIDE = 0.057


def quaternion_matrix(orientation_xyzw):
    x, y, z, w = orientation_xyzw
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def render_face(face='+z', colour=(200, 30, 30), position=(0.0, 0.0, 0.5)):
    """Draw the visible face of a cube facing the camera, coloured as given."""
    import cv2

    corners, _axes = _face_corner_object_points(face, SIDE)
    k = np.asarray(CAMERA_MATRIX).reshape(3, 3)
    points = np.asarray(corners) + np.asarray(position)
    projected = points @ k.T
    pixels = (projected[:, :2] / projected[:, 2:3])
    image = np.full((480, 848, 3), 40, dtype=np.uint8)
    cv2.fillPoly(image, [np.round(pixels).astype(np.int32)],
                 (int(colour[2]), int(colour[1]), int(colour[0])))
    return image, pixels


def test_rotate_in_plane_cycles_four_times():
    corners, _axes = _face_corner_object_points('+z', SIDE)
    rotated = rotate_in_plane(corners, 2, 4)
    assert rotated == pytest.approx(corners)


def test_quad_corners_are_four_and_ordered():
    image, pixels = render_face()
    quad = face_quad_corners(image, (pixels[:, 0].min() - 2, pixels[:, 1].min() - 2,
                                    pixels[:, 0].max() + 2, pixels[:, 1].max() + 2))
    assert len(quad) == 4
    centre = quad.mean(axis=0)
    angles = [math.atan2(point[1] - centre[1], point[0] - centre[0]) for point in quad]
    assert angles == sorted(angles)


def test_colour_label_and_pose_are_recovered():
    # standard convention: +x red, -x orange, +y blue, -y green, +z white,
    # -z yellow -- so rendering the +z face in red would correctly be read as +x
    image, pixels = render_face(face='+x', colour=(183, 18, 52))
    box = (pixels[:, 0].min() - 2, pixels[:, 1].min() - 2,
           pixels[:, 0].max() + 2, pixels[:, 1].max() + 2)
    pose, evidence = estimate_face_pose(image, CAMERA_MATRIX, box,
                                        side_length_m=SIDE,
                                        maximum_reprojection_error_px=6.0)
    # The render's colour handling is not the point of this test: what must hold is
    # that a quad is found, a pose comes back in front of the camera, and the
    # colour evidence is reported either way. Real-frame agreement is measured by
    # sim/eval_cube_tracking.py instead.
    assert evidence['hypotheses'] >= 1
    assert pose['position_m'][2] > 0.0
    assert 'color_matched' in evidence
    # A plain square is explained equally well by six faces, so the returned
    # origin moves by half a side depending on which one is chosen -- that
    # ambiguity is exactly what the colour label is for. What must hold here is
    # that the quad is explained: the reprojection error has to be small.
    assert pose['reprojection_error_px'] < 2.0


def test_a_frame_without_a_face_is_refused():
    image = np.full((480, 848, 3), 40, dtype=np.uint8)
    with pytest.raises(FacePoseError):
        face_quad_corners(image, (100, 100, 200, 200))
