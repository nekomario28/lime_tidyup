"""Tests for the pose evidence assembled from image, pose and geometry."""

import unittest
from types import SimpleNamespace

import numpy as np

from barcode_detector.cube_pose_3d import (
    CubePose3DError,
    estimate_cube_pose_3d,
)
from barcode_detector.cube_pose_evidence import (
    UNRESOLVED_COLOR_REASON,
    attach_pose_evidence,
)
from test_cube_face_colors import (
    CAMERA_MATRIX,
    FACE_TO_LABEL,
    PAINTED,
    SIDE_M,
    THREE_FACE_ORIENTATION,
    THREE_FACE_POSITION_M,
    keypoints_for_view,
    render_faces,
)


def camera_info():
    return SimpleNamespace(
        k=CAMERA_MATRIX,
        d=(),
        header=SimpleNamespace(frame_id='cube8_camera_optical'),
    )


def hidden_indices():
    from barcode_detector.cube_visibility import visible_vertex_mask

    mask = visible_vertex_mask(
        THREE_FACE_POSITION_M, THREE_FACE_ORIENTATION, side_length_m=SIDE_M
    )
    return {index for index, visible in enumerate(mask) if not visible}


def detection_for(confidences=None, keypoints=None):
    points = keypoints_for_view() if keypoints is None else keypoints
    if confidences is None:
        # A detector cannot measure a corner geometry says is hidden.
        hidden = hidden_indices()
        confidences = [
            0.0 if index in hidden else 0.9 for index in range(len(points))
        ]
    detection = {
        'class_name': '3-face-cube',
        'confidence': 0.9,
        'box_xyxy': [0.0, 0.0, 1.0, 1.0],
        'keypoints_xy': [list(point) for point in points],
        'keypoints_confidence': list(confidences),
        'pose_3d_debug': {'status': 'accepted'},
    }
    try:
        pose = estimate_cube_pose_3d(
            points,
            confidences,
            CAMERA_MATRIX,
            side_length_m=SIDE_M,
        )
    except CubePose3DError as error:
        # A corrupted observation set legitimately has no pose; the node would
        # publish the detection with the rejection reason instead.
        detection['pose_3d_debug'] = {
            'status': 'rejected', 'reason': str(error),
        }
    else:
        detection['pose_3d'] = pose.as_dict(
            frame_id='cube8_camera_optical',
            timestamp=1.0,
            side_length_m=SIDE_M,
        )
    return detection


def attach(detection, image, **overrides):
    arguments = dict(
        side_length_m=SIDE_M,
        frame_id='cube8_camera_optical',
        timestamp=1.0,
    )
    arguments.update(overrides)
    return attach_pose_evidence(
        image, camera_info(), detection, **arguments
    )


class DefaultsAreInertTest(unittest.TestCase):
    """Opt-in stages must not perturb the existing contract."""

    def test_both_stages_off_returns_the_detection_untouched(self):
        detection = detection_for()
        result = attach(detection, render_faces(PAINTED))
        self.assertNotIn('pose_evidence', result)
        self.assertEqual(
            sorted(result['pose_3d']), sorted(detection['pose_3d'])
        )
        self.assertNotIn(
            'color_fixed_semantics_resolved', result['pose_3d']
        )


class VisibilityGateTest(unittest.TestCase):
    """Visibility evidence decides whether a pose survives at all."""

    def test_consistent_observations_are_accepted(self):
        detection = detection_for()
        result = attach(
            detection,
            render_faces(PAINTED),
            enable_visibility_gate=True,
        )
        self.assertEqual(result['pose_evidence']['visibility']['status'],
                         'accepted')
        self.assertIn('pose_3d', result)
        self.assertEqual(
            result['pose_3d']['visibility_excluded_indices'], []
        )

    def test_contradictory_observations_remove_the_pose(self):
        keypoints = keypoints_for_view()
        # Displace the corner geometry says is hidden, twice over, so no single
        # omission can make the observation set self-consistent.
        corrupted = [list(point) for point in keypoints]
        for index in (7, 3):
            corrupted[index][0] += 12.0
            corrupted[index][1] += 9.0
        detection = detection_for(keypoints=corrupted)
        result = attach(
            detection,
            render_faces(PAINTED),
            enable_visibility_gate=True,
        )
        self.assertNotIn('pose_3d', result)
        self.assertEqual(result['pose_evidence']['visibility']['status'],
                         'rejected')


class ColorSelectionTest(unittest.TestCase):
    """Colour evidence is what sets the resolved flag, and only then."""

    def test_painted_faces_resolve_the_pose(self):
        detection = detection_for()
        result = attach(
            detection,
            render_faces(PAINTED),
            enable_face_color_selection=True,
        )
        pose = result['pose_3d']
        self.assertTrue(pose['color_fixed_semantics_resolved'])
        self.assertEqual(result['pose_evidence']['color']['status'],
                         'resolved')
        self.assertEqual(pose['hypothesis_color_margin'], 3)
        self.assertEqual(
            pose['visible_face_colors'],
            {slot: FACE_TO_LABEL[slot]
             for slot in pose['visible_face_colors']},
        )
        self.assertEqual(result['pose_evidence']['color']['hypothesis_count'],
                         3)

    def test_unpainted_faces_leave_the_pose_unresolved(self):
        blank = np.full((480, 640, 3), 128, dtype=np.uint8)
        detection = detection_for()
        result = attach(
            detection,
            blank,
            enable_face_color_selection=True,
        )
        pose = result['pose_3d']
        self.assertFalse(pose['color_fixed_semantics_resolved'])
        self.assertEqual(result['pose_evidence']['color']['status'],
                         'unresolved')
        self.assertEqual(result['pose_evidence']['color']['reason'],
                         UNRESOLVED_COLOR_REASON)
        # The pose itself is not thrown away: only the semantics stay unknown.
        self.assertIn('position_m', pose)

    def test_resolved_pose_keeps_the_base_position_when_it_is_the_match(self):
        detection = detection_for()
        result = attach(
            detection,
            render_faces(PAINTED),
            enable_face_color_selection=True,
        )
        for got, want in zip(
            result['pose_3d']['position_m'], THREE_FACE_POSITION_M
        ):
            self.assertAlmostEqual(got, want, places=6)

    def test_both_stages_together_resolve(self):
        detection = detection_for()
        result = attach(
            detection,
            render_faces(PAINTED),
            enable_visibility_gate=True,
            enable_face_color_selection=True,
        )
        self.assertEqual(result['pose_evidence']['visibility']['status'],
                         'accepted')
        self.assertTrue(result['pose_3d']['color_fixed_semantics_resolved'])
        self.assertEqual(
            sorted(result['pose_evidence']), ['color', 'visibility']
        )


if __name__ == '__main__':
    unittest.main()
