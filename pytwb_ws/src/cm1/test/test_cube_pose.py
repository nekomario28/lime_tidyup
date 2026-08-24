"""Tests for selecting optional 3D-pose distance evidence."""

import unittest

from cm1.lib.cube_pose import accepted_pose_distance


def detection_with_pose(**overrides):
    """Build accepted pose evidence with focused overrides."""
    pose = {
        'geometry_id': 'cube-axis-v1',
        'frame_id': 'camera_color_optical_frame',
        'timestamp': 100.0,
        'position_m': [0.01, -0.02, 0.55],
        'reprojection_error_px': 1.5,
        'correspondence_count': 8,
        'inlier_count': 7,
    }
    pose.update(overrides)
    return {'pose_3d': pose}


class AcceptedPoseDistanceTests(unittest.TestCase):
    """Verify pose selection never blocks the existing depth fallback."""

    def test_returns_camera_forward_z(self):
        distance = accepted_pose_distance(
            detection_with_pose(),
            now=101.0,
        )

        self.assertEqual(distance, 0.55)

    def test_rejects_stale_pose(self):
        distance = accepted_pose_distance(
            detection_with_pose(),
            now=104.0,
        )

        self.assertIsNone(distance)

    def test_rejects_low_quality_pose(self):
        distance = accepted_pose_distance(
            detection_with_pose(reprojection_error_px=5.0),
            now=101.0,
        )

        self.assertIsNone(distance)

    def test_allows_depth_fallback_when_pose_missing(self):
        self.assertIsNone(accepted_pose_distance({}, now=101.0))


if __name__ == '__main__':
    unittest.main()
