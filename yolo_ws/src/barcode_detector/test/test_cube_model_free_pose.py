"""Tests for the model-free (geometry + colour) pose route.

The fixtures are two frames of the generated 57 mm pose cube rendered in Gazebo
from the same pose (cube centre at 0.35 m, one corner towards the lens):

* ``frame_fill_light.png`` — dark backdrop with a fill light, so the white face
  is separable from grey;
* ``frame_single_sun.png`` — the inherited grey world with a single sun, where
  the white face shades towards grey.

Both are the frames the measurements in
``doc/research/2026-10-19-*.md`` were taken on, so the tests pin observed
behaviour rather than an idealised drawing. The detector box is the one the
shipped YOLO model produced on these frames.
"""

import math
import unittest
from pathlib import Path

import cv2
import numpy as np

from barcode_detector.cube_model_free_pose import (
    ModelFreePoseError,
    estimate_model_free_pose,
)

FIXTURES = Path(__file__).parent / 'fixtures' / 'cube_model_free'
CAMERA_MATRIX = (
    1060.9338813153618, 0.0, 640.5,
    0.0, 1060.9338813153618, 360.5,
    0.0, 0.0, 1.0,
)
DETECTOR_BOX = [492.0, 223.0, 791.0, 512.0]
TRUTH_POSITION = (0.0, 0.0, 0.35)


def _frame(name):
    image = cv2.imread(str(FIXTURES / name))
    if image is None:
        raise unittest.SkipTest(f'fixture missing: {name}')
    return image


class ModelFreePoseTest(unittest.TestCase):
    """The route must produce a colour-fixed pose, or refuse legibly."""

    def test_a_lit_view_resolves_the_pose_and_the_colours(self):
        pose = estimate_model_free_pose(
            _frame('frame_fill_light.png'), CAMERA_MATRIX, DETECTOR_BOX
        )
        self.assertTrue(pose.color_fixed_semantics_resolved)
        self.assertLess(pose.reprojection_error_px, 4.0)
        error_mm = math.sqrt(sum(
            (got - want) ** 2
            for got, want in zip(pose.translation_m, TRUTH_POSITION)
        )) * 1000.0
        self.assertLess(error_mm, 5.0, f'translation error {error_mm:.2f} mm')
        self.assertEqual(pose.correspondence_count, 6)
        self.assertEqual(pose.hypothesis_count, 3)
        # Colour is resolved by the labelled face map, so there is no competing
        # hypothesis left to score and no margin is claimed.
        self.assertIsNone(pose.color_margin)
        self.assertEqual(pose.color_agreement, 3)
        self.assertEqual(
            set(pose.visible_face_colors.values()), {'orange', 'blue', 'white'}
        )

    def test_a_shaded_white_face_keeps_the_pose_unresolved(self):
        pose = estimate_model_free_pose(
            _frame('frame_single_sun.png'), CAMERA_MATRIX, DETECTOR_BOX
        )
        self.assertFalse(pose.color_fixed_semantics_resolved)
        self.assertTrue(pose.unresolved_reason)
        self.assertIn(None, pose.visible_face_colors.values())
        # Geometry is still usable evidence even when colour abstains.
        self.assertLess(pose.reprojection_error_px, 4.0)

    def test_the_pose_is_deterministic(self):
        image = _frame('frame_fill_light.png')
        first = estimate_model_free_pose(image, CAMERA_MATRIX, DETECTOR_BOX)
        second = estimate_model_free_pose(image, CAMERA_MATRIX, DETECTOR_BOX)
        self.assertEqual(first.translation_m, second.translation_m)
        self.assertEqual(first.orientation_xyzw, second.orientation_xyzw)
        self.assertEqual(first.label_swaps, second.label_swaps)
        self.assertEqual(first.visible_face_colors, second.visible_face_colors)

    def test_ambiguity_is_reported_instead_of_a_pose_being_chosen(self):
        """Two relabellings fitting means handedness is undetermined."""
        from barcode_detector.cube8_pnp import CameraModel
        from barcode_detector.cube8_two_face_color_frontend import (
            recover_two_face_view_local_from_bgr,
        )
        from barcode_detector.cube8_view_local_pnp import (
            Cube8PnpError,
            estimate_view_local_pose_with_label_search,
        )

        image = _frame('frame_fill_light.png')
        observation = recover_two_face_view_local_from_bgr(
            image, DETECTOR_BOX
        )
        camera = CameraModel(
            k=CAMERA_MATRIX, d=(0.0, 0.0, 0.0, 0.0, 0.0)
        )
        points = observation.geometry.points
        # The 4 px gate leaves one relabelling; a 40 px gate also admits the
        # mirrored one, which must be refused rather than picked.
        with self.assertRaises(Cube8PnpError) as caught:
            estimate_view_local_pose_with_label_search(
                points, camera, side_length_m=0.057,
                maximum_reprojection_error_px=40.0,
            )
        self.assertIn('ambiguous', str(caught.exception))

    def test_silhouette_snapping_pulls_inset_corners_onto_the_outline(self):
        """The colour quads are inset; snapping must close that gap."""
        from barcode_detector.cube8_two_face_color_frontend import (
            TwoFaceColorFrontendConfig,
            snap_quads_to_silhouette,
        )

        outline = np.array(
            [[120, 90], [280, 90], [345, 200], [280, 310], [120, 310], [55, 200]],
            dtype=np.int32,
        )
        frame = np.full((400, 400, 3), 40, dtype=np.uint8)
        cv2.fillPoly(frame, [outline], (200, 200, 200))
        search_box = (40.0, 70.0, 360.0, 330.0)
        config = TwoFaceColorFrontendConfig()
        centre = np.array([200.0, 200.0])
        inset = 9.0
        quad = []
        for target in outline[:4]:
            direction = np.array(target, dtype=float) - centre
            quad.append(tuple(np.array(target, dtype=float)
                              - direction / np.linalg.norm(direction) * inset))
        diagonal = math.hypot(search_box[2] - search_box[0],
                              search_box[3] - search_box[1])
        snapped = snap_quads_to_silhouette(
            frame, {'search_box': search_box, 'vertices': {'y': tuple(quad)}},
            config, diagonal,
        )
        self.assertIsNotNone(snapped)

        def worst(vertices):
            return max(
                min(math.hypot(v[0] - t[0], v[1] - t[1]) for t in outline)
                for v in vertices
            )

        self.assertLess(worst(snapped['y']), worst(quad))
        self.assertLess(worst(snapped['y']), 3.0)

    def test_an_empty_frame_is_refused(self):
        image = _frame('frame_fill_light.png') * 0
        with self.assertRaises(ModelFreePoseError):
            estimate_model_free_pose(image, CAMERA_MATRIX, DETECTOR_BOX)

    def test_a_degenerate_bbox_is_refused(self):
        image = _frame('frame_fill_light.png')
        with self.assertRaises(Exception):
            estimate_model_free_pose(
                image, CAMERA_MATRIX, [10.0, 10.0, 10.0, 10.0]
            )

    def test_a_box_off_the_cube_is_refused(self):
        image = _frame('frame_fill_light.png')
        with self.assertRaises(ModelFreePoseError):
            estimate_model_free_pose(
                image, CAMERA_MATRIX, [20.0, 20.0, 200.0, 200.0]
            )


if __name__ == '__main__':
    unittest.main()
