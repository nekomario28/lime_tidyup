"""Tests for robust bounding-box depth evidence."""

import unittest

import numpy as np

from cm1.lib.cube_depth import (
    CubeDepthError,
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


if __name__ == '__main__':
    unittest.main()
