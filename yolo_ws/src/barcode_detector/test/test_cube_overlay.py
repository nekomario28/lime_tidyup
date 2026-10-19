"""Tests for the perception overlay that rqt shows."""

import math
import unittest

import numpy as np

from barcode_detector.cube_overlay import (
    OverlayError,
    draw_cube_overlay,
    overlay_is_blank,
)
from barcode_detector.cube_pose_3d import _cube_vertices  # noqa: PLC2701

CAMERA_MATRIX = (1060.9338813153618, 0.0, 640.5,
                 0.0, 1060.9338813153618, 360.5,
                 0.0, 0.0, 1.0)
SIDE_M = 0.057
POSE = {
    'position_m': [0.0, 0.0, 0.35],
    'orientation_xyzw': [0.0, 0.0, 0.0, 1.0],
}


def frame():
    return np.full((720, 1280, 3), 30, dtype=np.uint8)


def non_background_pixels(image):
    """Count pixels that are not the flat backdrop."""
    backdrop = image[0, 0]
    difference = np.abs(image.astype(np.int16) - backdrop.astype(np.int16))
    return int((difference.sum(axis=2) > 12).sum())


class OverlayDrawingTest(unittest.TestCase):
    """The overlay must draw evidence, and never claim more than it has."""

    def test_a_rejected_frame_still_gets_a_status_and_a_reason(self):
        annotated = draw_cube_overlay(
            frame(), status='rejected',
            reason='bbox source rejected: no depth blob',
        )
        self.assertFalse(overlay_is_blank(annotated))
        self.assertGreater(non_background_pixels(annotated), 200)

    def test_an_accepted_pose_draws_the_wireframe_where_the_pose_is(self):
        annotated = draw_cube_overlay(
            frame(), camera_matrix=CAMERA_MATRIX, pose=POSE,
            status='accepted', reprojection_error_px=1.03,
            color_resolved=True, label_swaps=(False, False, False),
            bbox_xyxy=(492.0, 223.0, 791.0, 512.0),
            bbox_source='depth_blob', bbox_quality=0.7,
        )
        # the drawn wireframe must sit inside the region the pose projects to
        projected = [
            _project_vertex(vertex, POSE, CAMERA_MATRIX)
            for vertex in _cube_vertices(SIDE_M)
        ]
        xs = [p[0] for p in projected]
        ys = [p[1] for p in projected]
        difference = np.abs(
            annotated.astype(np.int16) - frame().astype(np.int16)
        ).sum(axis=2)
        rows, columns = np.where(difference > 12)
        marked = [
            (column, row)
            for row, column in zip(rows, columns)
            if min(xs) - 6 <= column <= max(xs) + 6
            and min(ys) - 6 <= row <= max(ys) + 6
        ]
        self.assertGreater(len(marked), 50,
                           'no wireframe pixels inside the projected cube')

    def test_a_wrong_pose_is_drawn_somewhere_else(self):
        """The overlay must be able to fail: a displaced pose misses the cube."""
        correct = [
            _project_vertex(vertex, POSE, CAMERA_MATRIX)
            for vertex in _cube_vertices(SIDE_M)
        ]
        displaced = dict(POSE)
        displaced['position_m'] = [0.25, 0.0, 0.35]
        moved = [
            _project_vertex(vertex, displaced, CAMERA_MATRIX)
            for vertex in _cube_vertices(SIDE_M)
        ]
        separation = min(
            math.hypot(a[0] - b[0], a[1] - b[1])
            for a in correct for b in moved
        )
        self.assertGreater(
            separation, 50.0,
            'a displaced pose must land far from the true wireframe',
        )

    def test_face_regions_and_colour_labels_are_drawn(self):
        quad = [[600.0, 200.0], [700.0, 210.0], [690.0, 300.0],
                [590.0, 290.0]]
        annotated = draw_cube_overlay(
            frame(), face_regions={'+z': quad},
            visible_faces=('+z',), observed_colors={'+z': 'white'},
            status='unresolved',
            reason='visible face colours are not uniquely resolved',
        )
        # the quad outline is drawn in the face colour
        self.assertTrue(
            np.any(np.all(annotated == np.array([0, 255, 255], dtype=np.uint8),
                          axis=2)),
            'the face outline was not drawn',
        )
        self.assertGreater(non_background_pixels(annotated), 300)

    def test_the_bbox_and_its_source_are_drawn(self):
        annotated = draw_cube_overlay(
            frame(), bbox_xyxy=(492.0, 223.0, 791.0, 512.0),
            bbox_source='depth_blob', bbox_quality=0.7,
            status='rejected', reason='two-face frontend rejected',
        )
        self.assertTrue(
            np.any(np.all(annotated == np.array([255, 190, 0], dtype=np.uint8),
                          axis=2)),
            'the bbox was not drawn',
        )

    def test_absent_evidence_is_simply_absent(self):
        annotated = draw_cube_overlay(frame(), status='searching')
        self.assertFalse(overlay_is_blank(annotated))
        self.assertEqual(annotated.shape, frame().shape)

    def test_the_result_is_deterministic(self):
        first = draw_cube_overlay(
            frame(), camera_matrix=CAMERA_MATRIX, pose=POSE,
            status='accepted', bbox_xyxy=(1.0, 1.0, 100.0, 100.0),
        )
        second = draw_cube_overlay(
            frame(), camera_matrix=CAMERA_MATRIX, pose=POSE,
            status='accepted', bbox_xyxy=(1.0, 1.0, 100.0, 100.0),
        )
        self.assertTrue(np.array_equal(first, second))

    def test_the_input_image_is_not_modified(self):
        original = frame()
        before = original.copy()
        draw_cube_overlay(original, status='accepted')
        self.assertTrue(np.array_equal(original, before))

    def test_invalid_input_is_an_error(self):
        with self.assertRaises(OverlayError):
            draw_cube_overlay(None, status='accepted')
        with self.assertRaises(OverlayError):
            draw_cube_overlay(np.zeros((10, 10), dtype=np.uint8))


def _project_vertex(vertex, pose, camera_matrix):
    """Project one object-frame vertex, for the test's own expectation."""
    from barcode_detector.cube_overlay import _project

    point = (vertex.x, vertex.y, vertex.z) if hasattr(vertex, 'x') else vertex
    return _project(
        point, pose['position_m'], pose['orientation_xyzw'], camera_matrix,
    )


if __name__ == '__main__':
    unittest.main()
