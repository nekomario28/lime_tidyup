"""Tests for robust bounding-box depth evidence."""

import unittest

import numpy as np

from cm1.lib.cube_depth import (
    CubeDepthError,
    CubeTemporalGate,
    check_known_size_geometry,
    consistent_depth_median,
    estimate_cube_depth,
)


class CubeDepthTests(unittest.TestCase):
    """Verify low-quality or inconsistent depth fails closed."""

    def test_uses_inner_box_median_in_millimetres(self):
        """Prefer the cube's inner ROI over surrounding background depth."""
        image = np.full((100, 100), 2000, dtype=np.uint16)
        image[35:65, 35:65] = 500

        estimate = estimate_cube_depth(
            image,
            (20, 20, 80, 80),
            encoding='16UC1',
            rgb_width=100,
            rgb_height=100,
            roi_fraction=0.50,
            minimum_valid_ratio=0.90,
            maximum_mad_m=0.01,
        )

        self.assertEqual(estimate.distance_m, 0.5)
        self.assertEqual(estimate.valid_ratio, 1.0)
        self.assertEqual(estimate.mad_m, 0.0)

    def test_rejects_sparse_depth_region(self):
        """Reject an ROI when too few pixels contain valid depth."""
        image = np.zeros((40, 40), dtype=np.uint16)
        image[19:21, 19:21] = 500

        with self.assertRaisesRegex(CubeDepthError, 'valid ratio'):
            estimate_cube_depth(
                image,
                (10, 10, 30, 30),
                encoding='16UC1',
                rgb_width=40,
                rgb_height=40,
                roi_fraction=0.50,
                minimum_valid_ratio=0.50,
            )

    def test_default_keeps_existing_eleven_pixel_center_patch(self):
        """Keep the prior 11x11 center-depth behavior for other trees."""
        image = np.full((100, 100), 500, dtype=np.uint16)

        estimate = estimate_cube_depth(
            image,
            (20, 20, 80, 80),
            encoding='16UC1',
            rgb_width=100,
            rgb_height=100,
        )

        self.assertEqual(estimate.roi_xyxy, (45, 45, 56, 56))
        self.assertEqual(estimate.sample_count, 121)

    def test_rejects_mixed_surface_depth(self):
        """Reject an ROI whose median absolute deviation is too large."""
        image = np.full((40, 40), 500, dtype=np.uint16)
        image[:, 20:] = 1000

        with self.assertRaisesRegex(CubeDepthError, 'MAD'):
            estimate_cube_depth(
                image,
                (0, 0, 40, 40),
                encoding='16UC1',
                rgb_width=40,
                rgb_height=40,
                roi_fraction=1.0,
                minimum_valid_ratio=1.0,
                maximum_mad_m=0.10,
            )

    def test_accepts_consistent_float_depth(self):
        """Accept a dense floating-point depth ROI measured in metres."""
        image = np.full((20, 20), 0.42, dtype=np.float32)
        image[0, 0] = np.nan

        estimate = estimate_cube_depth(
            image,
            (0, 0, 20, 20),
            encoding='32FC1',
            rgb_width=20,
            rgb_height=20,
            roi_fraction=0.50,
            minimum_valid_ratio=0.95,
            maximum_mad_m=0.01,
        )

        self.assertAlmostEqual(estimate.distance_m, 0.42, places=6)

    def test_repeated_samples_use_median(self):
        """Use the median when repeated frame estimates agree."""
        distance = consistent_depth_median(
            (0.50, 0.51, 0.49),
            maximum_spread_m=0.03,
        )

        self.assertEqual(distance, 0.50)

    def test_repeated_samples_reject_large_spread(self):
        """Reject motion when repeated frame estimates disagree."""
        with self.assertRaisesRegex(CubeDepthError, 'spread'):
            consistent_depth_median(
                (0.50, 0.56, 0.49),
                maximum_spread_m=0.03,
            )

    def test_known_size_geometry_accepts_loose_cube_projection(self):
        """Accept an observed extent inside the loose pose-aware range."""
        result = check_known_size_geometry(
            (100, 100, 157, 153),
            distance_m=0.60,
            focal_length_px=600.0,
            side_length_m=0.057,
            minimum_scale=0.60,
            maximum_scale=1.90,
        )

        self.assertTrue(result.accepted)
        self.assertAlmostEqual(result.expected_pixel_size, 57.0)
        self.assertEqual(result.observed_pixel_size, 57.0)

    def test_known_size_geometry_rejects_implausibly_large_box(self):
        """Reject a box far larger than a 57 mm cube at the measured depth."""
        result = check_known_size_geometry(
            (100, 100, 268, 250),
            distance_m=0.60,
            focal_length_px=600.0,
        )

        self.assertFalse(result.accepted)
        self.assertGreater(
            result.observed_pixel_size,
            result.maximum_pixel_size,
        )

    def test_known_size_geometry_rejects_missing_camera_calibration(self):
        """Fail closed when focal length is unavailable."""
        with self.assertRaisesRegex(CubeDepthError, 'focal length'):
            check_known_size_geometry(
                (100, 100, 157, 157),
                distance_m=0.60,
                focal_length_px=0.0,
            )

    def test_temporal_gate_confirms_three_of_five_nearby_candidates(self):
        """Confirm after three valid observations from one nearby track."""
        gate = CubeTemporalGate(
            window_size=5,
            required_count=3,
            maximum_center_distance_px=20.0,
        )

        states = [
            gate.update((100, 100), valid=True),
            gate.update((103, 101), valid=False),
            gate.update((105, 102), valid=True),
            gate.update((108, 104), valid=True),
        ]

        self.assertFalse(states[2].confirmed)
        self.assertTrue(states[3].confirmed)
        self.assertEqual(states[3].valid_count, 3)
        self.assertEqual(states[3].window_size, 5)

    def test_temporal_gate_resets_for_a_distant_candidate(self):
        """Do not combine observations whose centers imply another cube."""
        gate = CubeTemporalGate(
            window_size=5,
            required_count=3,
            maximum_center_distance_px=20.0,
        )
        gate.update((100, 100), valid=True)
        gate.update((105, 100), valid=True)

        state = gate.update((200, 200), valid=True)

        self.assertTrue(state.track_reset)
        self.assertFalse(state.confirmed)
        self.assertEqual(state.valid_count, 1)

    def test_temporal_gate_keeps_invalid_frames_unconfirmed(self):
        """Keep 2D/missing-depth candidates without granting authority."""
        gate = CubeTemporalGate(window_size=5, required_count=3)

        state = None
        for center in ((100, 100), (101, 100), None, (102, 101), None):
            state = gate.update(center, valid=False)

        self.assertFalse(state.confirmed)
        self.assertEqual(state.valid_count, 0)

    def test_temporal_gate_rejects_impossible_threshold(self):
        """Reject K values that cannot fit inside N."""
        with self.assertRaisesRegex(CubeDepthError, 'within the window'):
            CubeTemporalGate(window_size=5, required_count=6)


if __name__ == '__main__':
    unittest.main()
