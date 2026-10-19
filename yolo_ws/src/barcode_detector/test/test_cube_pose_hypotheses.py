"""Tests for symmetry-equivalent cube poses and colour-based selection."""

import unittest

from barcode_detector.cube_pose_3d import CubePose3DError
from barcode_detector.cube_pose_hypotheses import (
    STANDARD_FACE_COLORS,
    enumerate_pose_hypotheses,
    rotation_angle_deg,
    select_by_face_colors,
)
from test_cube_visibility import (
    CAMERA_MATRIX,
    EXPECTED_HIDDEN,
    _three_face_keypoints,
)


def _confidences(hidden_corners):
    return tuple(
        0.0 if index in hidden_corners else 0.9 for index in range(8)
    )


class HypothesisEnumerationTest(unittest.TestCase):
    """The ambiguity is structural: enumerate it, do not solve it away."""

    def test_three_face_view_yields_three_poses(self):
        hypotheses = enumerate_pose_hypotheses(
            _three_face_keypoints(),
            _confidences({EXPECTED_HIDDEN}),
            CAMERA_MATRIX,
        )
        self.assertEqual(len(hypotheses), 3)
        base = hypotheses[0]
        angles = sorted(
            round(rotation_angle_deg(base.pose[1], item.pose[1]), 3)
            for item in hypotheses
        )
        self.assertEqual(angles, [0.0, 120.0, 120.0])

    def test_each_hypothesis_predicts_a_unique_assignment(self):
        hypotheses = enumerate_pose_hypotheses(
            _three_face_keypoints(),
            _confidences({EXPECTED_HIDDEN}),
            CAMERA_MATRIX,
        )
        assignments = {item.color_assignment() for item in hypotheses}
        self.assertEqual(len(assignments), 3)
        for item in hypotheses:
            colors = item.predicted_colors
            self.assertEqual(len(colors), 3)
            self.assertEqual(len(set(colors)), 3)

    def test_frontal_view_is_rejected_before_the_ambiguity_matters(self):
        # Four coplanar corners do not determine a pose, so the base solve
        # fails closed. The four-fold colour ambiguity is therefore never
        # reached at runtime; the geometric count stays documented in the
        # research note.
        with self.assertRaises(CubePose3DError):
            enumerate_pose_hypotheses(
                _frontal_keypoints(),
                _confidences({4, 5, 6, 7}),
                CAMERA_MATRIX,
            )

    def test_enumeration_is_deterministic(self):
        first = enumerate_pose_hypotheses(
            _three_face_keypoints(),
            _confidences({EXPECTED_HIDDEN}),
            CAMERA_MATRIX,
        )
        second = enumerate_pose_hypotheses(
            _three_face_keypoints(),
            _confidences({EXPECTED_HIDDEN}),
            CAMERA_MATRIX,
        )
        self.assertEqual(
            [item.pose for item in first], [item.pose for item in second]
        )

    def test_custom_colour_labels_are_used(self):
        palette = {name: f'c{index}' for index, name in
                   enumerate(STANDARD_FACE_COLORS)}
        hypotheses = enumerate_pose_hypotheses(
            _three_face_keypoints(),
            _confidences({EXPECTED_HIDDEN}),
            CAMERA_MATRIX,
            face_colors=palette,
        )
        for item in hypotheses:
            for color in item.predicted_colors:
                self.assertTrue(color.startswith('c'))

    def test_missing_palette_entries_are_an_error(self):
        with self.assertRaises(CubePose3DError):
            enumerate_pose_hypotheses(
                _three_face_keypoints(),
                _confidences({EXPECTED_HIDDEN}),
                CAMERA_MATRIX,
                face_colors={'+x': 'red'},
            )


class ColourSelectionTest(unittest.TestCase):
    """Selection uses predicted-versus-observed colour, not preference."""

    def setUp(self):
        self.hypotheses = enumerate_pose_hypotheses(
            _three_face_keypoints(),
            _confidences({EXPECTED_HIDDEN}),
            CAMERA_MATRIX,
        )

    def test_the_matching_assignment_wins(self):
        for target in self.hypotheses:
            observed = dict(target.color_assignment())
            selection = select_by_face_colors(self.hypotheses, observed)
            self.assertIsNotNone(selection)
            self.assertEqual(
                selection.hypothesis.color_assignment(),
                target.color_assignment(),
            )
            self.assertEqual(selection.agreement, 3)
            self.assertEqual(selection.margin, 3)

    def test_a_partial_observation_is_missing_evidence_not_a_match(self):
        observed = dict(self.hypotheses[0].color_assignment())
        observed.pop('+x')
        with self.assertRaises(CubePose3DError):
            select_by_face_colors(self.hypotheses, observed)

    def test_an_unlabelled_face_selects_nothing(self):
        observed = dict(self.hypotheses[0].color_assignment())
        observed[self.hypotheses[0].visible_slots[0]] = None
        self.assertIsNone(select_by_face_colors(self.hypotheses, observed))

    def test_unmatched_colours_select_nothing(self):
        slots = self.hypotheses[0].visible_slots
        observed = {slot: 'unknown' for slot in slots}
        self.assertIsNone(select_by_face_colors(self.hypotheses, observed))

    def test_identical_predictions_select_nothing(self):
        # When two hypotheses predict the same colours, agreement cannot break
        # the tie and the selector must return nothing rather than pick one.
        hypotheses = self.hypotheses
        observed = dict(hypotheses[0].color_assignment())
        tied = list(hypotheses) + [hypotheses[0]]
        self.assertIsNone(select_by_face_colors(tied, observed))


def _frontal_keypoints():
    """Project the eight vertices for a camera on the -z side."""
    from test_cube_visibility import _project

    position = (0.0, 0.0, 0.6)
    orientation = (0.0, 0.0, 0.0, 1.0)
    half = 0.057 / 2.0
    return [
        _project(
            tuple(sign * half for sign in signs),
            position,
            orientation,
        )
        for signs in (
            (-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
            (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1),
        )
    ]


if __name__ == '__main__':
    unittest.main()
