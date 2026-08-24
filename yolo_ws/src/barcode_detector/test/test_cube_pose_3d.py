"""Tests for the bounded Cube8 PnP contract."""

import unittest

from barcode_detector.cube_pose_3d import (
    CubePose3DError,
    PnpBackendResult,
    estimate_cube_pose_3d,
)


CAMERA_MATRIX = (
    600.0, 0.0, 320.0,
    0.0, 600.0, 240.0,
    0.0, 0.0, 1.0,
)

KEYPOINTS = (
    (100.0, 100.0),
    (140.0, 100.0),
    (140.0, 140.0),
    (100.0, 140.0),
    (110.0, 110.0),
    (150.0, 110.0),
    (150.0, 150.0),
    (110.0, 150.0),
)


class FakeBackend:
    """Return deterministic PnP evidence without an OpenCV dependency."""

    def __init__(self, *, projected_points=KEYPOINTS, inliers=range(8)):
        self.projected_points = projected_points
        self.inliers = tuple(inliers)

    def solve(
        self,
        object_points,
        image_points,
        camera_matrix,
        distortion,
        maximum_reprojection_error_px,
    ):
        return PnpBackendResult(
            success=True,
            rotation_vector=(0.0, 0.0, 0.0),
            translation_m=(0.02, -0.01, 0.50),
            inlier_indices=self.inliers,
            projected_points=tuple(self.projected_points),
            rotation_matrix=(
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
        )


class CubePose3DTests(unittest.TestCase):
    """Exercise accepted and rejected PnP evidence."""

    def test_estimate_cube_pose_accepts_eight_ordered_points(self):
        pose = estimate_cube_pose_3d(
            KEYPOINTS,
            (0.9,) * 8,
            CAMERA_MATRIX,
            backend=FakeBackend(),
        )

        self.assertEqual(pose.translation_m, (0.02, -0.01, 0.50))
        self.assertEqual(
            pose.orientation_xyzw,
            (0.0, 0.0, 0.0, 1.0)
        )
        self.assertEqual(pose.inlier_count, 8)
        self.assertEqual(pose.reprojection_error_px, 0.0)

    def test_estimate_cube_pose_rejects_too_few_confident_points(self):
        with self.assertRaisesRegex(CubePose3DError, 'insufficient'):
            estimate_cube_pose_3d(
                KEYPOINTS,
                (0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1),
                CAMERA_MATRIX,
                backend=FakeBackend(),
            )

    def test_estimate_cube_pose_rejects_large_reprojection_error(self):
        projected = tuple((x + 10.0, y) for x, y in KEYPOINTS)
        with self.assertRaisesRegex(CubePose3DError, 'reprojection'):
            estimate_cube_pose_3d(
                KEYPOINTS,
                (0.9,) * 8,
                CAMERA_MATRIX,
                backend=FakeBackend(projected_points=projected),
            )


if __name__ == '__main__':
    unittest.main()
