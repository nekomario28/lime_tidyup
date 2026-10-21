"""Tests for silhouette refinement of a prior cube pose."""
import math

import numpy as np
import pytest

from barcode_detector.cube_pose_tracking import (
    CubePoseTrackingError,
    colour_mask,
    project,
    quaternion_matrix,
    refine_pose_from_silhouette,
    silhouette_corners,
    vertices_object_frame,
)

CAMERA_MATRIX = (455.0, 0.0, 424.0, 0.0, 455.0, 240.0, 0.0, 0.0, 1.0)
SIDE = 0.057
IMAGE = (480, 848)


def rotation_z(degrees):
    half = math.radians(degrees) / 2.0
    return (0.0, 0.0, math.sin(half), math.cos(half))


def rotation_xz(x_degrees, z_degrees):
    """A general orientation: an axis-aligned cube only shows a planar square."""
    from pyquaternion import Quaternion

    q = Quaternion(axis=[1.0, 0.0, 0.0], degrees=x_degrees) * Quaternion(
        axis=[0.0, 0.0, 1.0], degrees=z_degrees)
    return (q.x, q.y, q.z, q.w)


def render_cube(position, orientation, *, mask_only=False):
    """Draw the cube's silhouette as a saturated region, like the colour blob."""
    import cv2

    image = np.full((IMAGE[0], IMAGE[1], 3), 40, dtype=np.uint8)
    rotation = quaternion_matrix(orientation)
    points = vertices_object_frame(SIDE) @ rotation.T + np.asarray(position)
    pixels = project(points, CAMERA_MATRIX)
    hull = cv2.convexHull(pixels.astype(np.float32)).reshape(-1, 2)
    cv2.fillPoly(image, [np.round(hull).astype(np.int32)], (0, 120, 255))
    return image


def test_silhouette_corners_recover_the_projected_hull():
    image = render_cube((0.0, 0.0, 0.6), rotation_xz(25.0, 15.0))
    mask, area = colour_mask(image)
    corners, _contour = silhouette_corners(mask)
    assert area > 1000
    assert 5 <= len(corners) <= 8


def test_refine_recovers_the_true_pose_from_a_perturbed_prior():
    true_position = (0.0, 0.0, 0.6)
    true_orientation = rotation_xz(25.0, 15.0)
    image = render_cube(true_position, true_orientation)

    prior_position = (0.010, 0.008, 0.620)
    prior_orientation = rotation_xz(20.0, 22.0)
    pose, evidence = refine_pose_from_silhouette(
        image, CAMERA_MATRIX, prior_position, prior_orientation,
        side_length_m=SIDE, maximum_corner_distance_px=25.0)

    error = math.dist(pose.translation_m, true_position)
    assert error < 0.02, (pose.translation_m, error)
    assert evidence['matched_vertices'] >= 4
    # the refinement must beat the prior it started from
    prior_error = math.dist(prior_position, true_position)
    assert error < prior_error


def test_a_hopeless_prior_is_refused_instead_of_answered():
    image = render_cube((0.0, 0.0, 0.6), rotation_xz(25.0, 15.0))
    with pytest.raises(CubePoseTrackingError):
        refine_pose_from_silhouette(
            image, CAMERA_MATRIX, (0.30, 0.0, 0.9), rotation_xz(0.0, 0.0),
            side_length_m=SIDE, maximum_corner_distance_px=8.0)


def test_a_frame_without_a_coloured_region_is_refused():
    image = np.full((IMAGE[0], IMAGE[1], 3), 40, dtype=np.uint8)
    with pytest.raises(CubePoseTrackingError):
        colour_mask(image)
