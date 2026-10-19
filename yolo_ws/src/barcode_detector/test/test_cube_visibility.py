"""Tests for geometric vertex visibility and its observation discipline."""

import math
import unittest

from barcode_detector.cube_pose_3d import estimate_cube_pose_3d
from barcode_detector.cube_visibility import (
    CANONICAL_VERTEX_SIGNS,
    VISIBILITY_CONTRADICTED,
    VISIBILITY_OBSERVED,
    VISIBILITY_OCCLUDED,
    VISIBILITY_UNMEASURED,
    classify_vertices,
    estimate_visibility_consistent_pose,
    hidden_vertex_indices,
    observation_mask,
    visible_vertex_mask,
)


SIDE_M = 0.057
CAMERA_MATRIX = (
    600.0, 0.0, 320.0,
    0.0, 600.0, 240.0,
    0.0, 0.0, 1.0,
)

# Camera in object frame at (+x, -y, -z) -> the +X, -Y and -Z faces are
# camera-facing, so every vertex but (-1, +1, +1) is visible.
THREE_FACE_POSITION_M = (-0.4, 0.35, 0.6)
THREE_FACE_ORIENTATION = (0.0, 0.0, 0.0, 1.0)
EXPECTED_HIDDEN = 7


def _project(vertex, position_m, orientation):
    x, y, z = vertex
    roll_x, roll_y, roll_z, w = orientation
    assert (roll_x, roll_y, roll_z, w) == (0.0, 0.0, 0.0, 1.0)
    camera = (
        x + position_m[0],
        y + position_m[1],
        z + position_m[2],
    )
    u = CAMERA_MATRIX[0] * camera[0] / camera[2] + CAMERA_MATRIX[2]
    v = CAMERA_MATRIX[4] * camera[1] / camera[2] + CAMERA_MATRIX[5]
    return (u, v)


def _three_face_keypoints():
    half = SIDE_M / 2.0
    return [
        _project(
            tuple(sign * half for sign in signs),
            THREE_FACE_POSITION_M,
            THREE_FACE_ORIENTATION,
        )
        for signs in CANONICAL_VERTEX_SIGNS
    ]


def _translation_error(pose, expected):
    return math.sqrt(
        sum(
            (got - want) ** 2
            for got, want in zip(pose.translation_m, expected)
        )
    )


class VisibleVertexMaskTest(unittest.TestCase):
    """Geometry decides visibility; the detector does not."""

    def test_face_on_view_sees_the_four_near_corners(self):
        mask = visible_vertex_mask((0.0, 0.0, 0.5), (0.0, 0.0, 0.0, 1.0))
        self.assertEqual(mask, (True, True, True, True, False, False,
                                False, False))

    def test_three_face_view_sees_seven_of_eight(self):
        mask = visible_vertex_mask(
            THREE_FACE_POSITION_M, THREE_FACE_ORIENTATION
        )
        self.assertEqual(sum(mask), 7)
        hidden = hidden_vertex_indices(
            THREE_FACE_POSITION_M, THREE_FACE_ORIENTATION
        )
        self.assertEqual(hidden, (EXPECTED_HIDDEN,))

    def test_visibility_follows_the_pose_not_the_lateral_position(self):
        # Moving the camera along +X without rotating exposes the +X face.
        near = visible_vertex_mask((0.0, 0.0, 0.5), (0.0, 0.0, 0.0, 1.0))
        shifted = visible_vertex_mask((0.08, 0.0, 0.5), (0.0, 0.0, 0.0, 1.0))
        self.assertEqual(sum(near), 4)
        self.assertGreater(sum(shifted), sum(near))

    def test_non_finite_geometry_is_an_error_not_a_mask(self):
        with self.assertRaises(ValueError):
            visible_vertex_mask(
                (float('nan'), 0.0, 0.5), (0.0, 0.0, 0.0, 1.0)
            )


class CanonicalOrderTest(unittest.TestCase):
    """The module's vertex order must be the PnP contract's vertex order."""

    def test_sign_order_matches_the_pnp_vertex_order(self):
        from barcode_detector.cube_pose_3d import _cube_vertices

        half = CANONICAL_VERTEX_SIGNS and 0.057 / 2.0
        expected = tuple(
            tuple(int(round(value / half)) for value in vertex)
            for vertex in _cube_vertices(0.057)
        )
        self.assertEqual(expected, CANONICAL_VERTEX_SIGNS)


class VertexClassificationTest(unittest.TestCase):
    """Contradictions stay visible instead of being smoothed away."""

    def test_states_separate_measured_visible_and_impossible(self):
        mask = (True, True, False, False)
        confidences = (0.9, 0.2, 0.9, 0.1)
        states = classify_vertices(confidences, mask)
        self.assertEqual(
            states,
            (
                VISIBILITY_OBSERVED,
                VISIBILITY_UNMEASURED,
                VISIBILITY_CONTRADICTED,
                VISIBILITY_OCCLUDED,
            ),
        )

    def test_only_observed_vertices_may_be_observations(self):
        mask = (True, True, False, False)
        confidences = (0.9, 0.2, 0.9, 0.1)
        self.assertEqual(
            observation_mask(confidences, mask),
            (True, False, False, False),
        )

    def test_mismatched_lengths_are_an_error(self):
        with self.assertRaises(ValueError):
            classify_vertices((0.9, 0.9), (True, True, True))


class VisibilityConsistentPoseTest(unittest.TestCase):
    """A pose must agree with the visibility it predicts."""

    def test_clean_observations_are_accepted(self):
        keypoints = _three_face_keypoints()
        # The hidden corner is not measured, as a real detector cannot see it.
        confidences = tuple(
            0.0 if index == EXPECTED_HIDDEN else 0.9
            for index in range(8)
        )
        result = estimate_visibility_consistent_pose(
            keypoints, confidences, CAMERA_MATRIX
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.violation_count, 0)
        self.assertEqual(result.excluded_indices, ())
        self.assertLess(
            _translation_error(result.pose, THREE_FACE_POSITION_M), 5e-3
        )

    def test_an_impossible_detection_is_dropped_not_averaged(self):
        keypoints = _three_face_keypoints()
        confidences = (0.9,) * 8
        # A confident detection of the corner geometry says is hidden. The
        # offset is small enough that the confidence-only path still accepts
        # the pose: this is the case a reprojection budget cannot catch.
        corrupted = list(keypoints)
        hidden_u, hidden_v = corrupted[EXPECTED_HIDDEN]
        corrupted[EXPECTED_HIDDEN] = (hidden_u + 6.0, hidden_v + 4.0)

        plain = estimate_cube_pose_3d(
            corrupted, confidences, CAMERA_MATRIX
        )
        consistent = estimate_visibility_consistent_pose(
            corrupted, confidences, CAMERA_MATRIX
        )

        self.assertIsNotNone(consistent)
        self.assertEqual(
            consistent.excluded_indices, (EXPECTED_HIDDEN,)
        )
        self.assertGreater(plain.reprojection_error_px, 1.0)
        self.assertGreater(
            _translation_error(plain, THREE_FACE_POSITION_M), 1e-4
        )
        self.assertLess(
            consistent.pose.reprojection_error_px, 1e-3
        )
        self.assertLess(
            _translation_error(consistent.pose, THREE_FACE_POSITION_M), 5e-3
        )

    def test_contradictions_without_a_consistent_candidate_fail_closed(self):
        keypoints = _three_face_keypoints()
        confidences = (0.9,) * 8
        corrupted = list(keypoints)
        # Corrupt the hidden corner and a visible one: excluding either still
        # leaves the other contradicted, so no candidate is admissible.
        for index in (EXPECTED_HIDDEN, 3):
            u, v = corrupted[index]
            corrupted[index] = (u + 12.0, v + 9.0)
        self.assertIsNone(
            estimate_visibility_consistent_pose(
                corrupted, confidences, CAMERA_MATRIX
            )
        )

    def test_too_few_confident_observations_fail_closed(self):
        keypoints = _three_face_keypoints()
        confidences = (0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1)
        self.assertIsNone(
            estimate_visibility_consistent_pose(
                keypoints, confidences, CAMERA_MATRIX
            )
        )

    def test_result_is_deterministic(self):
        keypoints = _three_face_keypoints()
        confidences = (0.9,) * 8
        first = estimate_visibility_consistent_pose(
            keypoints, confidences, CAMERA_MATRIX
        )
        second = estimate_visibility_consistent_pose(
            keypoints, confidences, CAMERA_MATRIX
        )
        self.assertEqual(first.pose.translation_m, second.pose.translation_m)
        self.assertEqual(
            first.pose.orientation_xyzw, second.pose.orientation_xyzw
        )
        self.assertEqual(first.visible_mask, second.visible_mask)


if __name__ == '__main__':
    unittest.main()
