"""Tests for the swappable cube bbox source."""

import unittest

import numpy as np

from barcode_detector.cube_bbox_source import (
    BboxSourceError,
    DepthBlobConfig,
    choose_bbox,
    depth_blob_bbox,
    detector_bbox,
)

HEIGHT = 480
WIDTH = 640


def depth_scene(*, cube_depth=0.45, cube_box=(260, 170, 380, 300),
                plane_depth=0.95, noise=0.002):
    """A cube-shaped blob on a plane, as a float32 depth image in metres."""
    rng = np.random.default_rng(3)
    depth = np.full((HEIGHT, WIDTH), plane_depth, dtype=np.float32)
    left, top, right, bottom = cube_box
    depth[top:bottom, left:right] = cube_depth
    depth += rng.normal(0.0, noise, depth.shape).astype(np.float32)
    depth[:20, :] = np.nan
    return depth


class DepthBlobTest(unittest.TestCase):
    """A compact near blob is accepted; everything else is refused."""

    def test_a_compact_near_blob_is_accepted(self):
        result = depth_blob_bbox(depth_scene())
        self.assertEqual(result.source, 'depth_blob')
        left, top, right, bottom = result.box_xyxy
        # the box must cover the blob, with the configured margin
        self.assertLess(left, 260)
        self.assertGreater(right, 380)
        self.assertLess(top, 170)
        self.assertGreater(bottom, 300)
        self.assertAlmostEqual(result.detail['median_depth_m'], 0.45, places=2)
        self.assertGreater(result.quality, 0.0)

    def test_a_far_plane_alone_is_refused(self):
        depth = np.full((HEIGHT, WIDTH), 0.95, dtype=np.float32)
        try:
            depth_blob_bbox(depth)
        except BboxSourceError as error:
            self.assertIn('band', str(error).lower())
        else:
            raise AssertionError('a plane inside the band must not be a cube')

    def test_a_plane_inside_the_band_is_refused_by_shape(self):
        depth = np.full((HEIGHT, WIDTH), 0.45, dtype=np.float32)
        with self.assertRaises(BboxSourceError):
            depth_blob_bbox(depth)

    def test_a_sloped_surface_is_refused(self):
        rows = np.linspace(0.30, 0.75, HEIGHT, dtype=np.float32)[:, None]
        depth = np.repeat(rows, WIDTH, axis=1)
        with self.assertRaises(BboxSourceError):
            depth_blob_bbox(depth)

    def test_the_nearest_blob_wins(self):
        depth = depth_scene()
        # a second, farther blob of the same size
        depth[60:150, 60:140] = 0.80
        result = depth_blob_bbox(depth)
        self.assertLess(result.detail['median_depth_m'], 0.60)

    def test_a_too_small_blob_is_refused(self):
        depth = depth_scene(cube_box=(300, 220, 308, 228))
        with self.assertRaises(BboxSourceError):
            depth_blob_bbox(depth)

    def test_invalid_input_is_an_error(self):
        with self.assertRaises(BboxSourceError):
            depth_blob_bbox(np.zeros((0, 0), dtype=np.float32))
        with self.assertRaises(BboxSourceError):
            depth_blob_bbox(np.zeros((HEIGHT, WIDTH)))
        with self.assertRaises(BboxSourceError):
            depth_blob_bbox(depth_scene(), roi_xyxy=(10, 10, 12, 12))

    def test_a_bad_configuration_is_rejected(self):
        with self.assertRaises(BboxSourceError):
            depth_blob_bbox(
                depth_scene(),
                config=DepthBlobConfig(minimum_range_m=0.9,
                                       maximum_range_m=0.2),
            )

    def test_the_result_is_deterministic(self):
        first = depth_blob_bbox(depth_scene())
        second = depth_blob_bbox(depth_scene())
        self.assertEqual(first.box_xyxy, second.box_xyxy)
        self.assertEqual(first.detail, second.detail)


class DetectorTest(unittest.TestCase):
    """The detector box stays available and validated."""

    def test_a_valid_box_passes_through(self):
        result = detector_bbox([492.0, 223.0, 791.0, 512.0])
        self.assertEqual(result.box_xyxy, (492.0, 223.0, 791.0, 512.0))
        self.assertEqual(result.source, 'detector')

    def test_a_degenerate_box_is_refused(self):
        with self.assertRaises(BboxSourceError):
            detector_bbox([10.0, 10.0, 10.0, 10.0])

    def test_a_non_finite_box_is_refused(self):
        with self.assertRaises(BboxSourceError):
            detector_bbox([0.0, 0.0, float('nan'), 10.0])


class DispatchTest(unittest.TestCase):
    """One entry point, explicit sources, no silent fallback."""

    def test_depth_dispatch_uses_the_depth_image(self):
        result = choose_bbox('depth_blob', depth_image=depth_scene())
        self.assertEqual(result.source, 'depth_blob')

    def test_detector_dispatch_uses_the_box(self):
        result = choose_bbox(
            'detector', detector_box=[10.0, 10.0, 60.0, 60.0]
        )
        self.assertEqual(result.source, 'detector')

    def test_a_missing_input_is_an_error_not_a_fallback(self):
        with self.assertRaises(BboxSourceError):
            choose_bbox('depth_blob')
        with self.assertRaises(BboxSourceError):
            choose_bbox('detector', depth_image=depth_scene())

    def test_the_colour_pair_experiment_is_not_a_production_source(self):
        with self.assertRaises(BboxSourceError) as caught:
            choose_bbox('colour_pair')
        self.assertIn('experiment', str(caught.exception))

    def test_an_unknown_source_is_an_error(self):
        with self.assertRaises(BboxSourceError):
            choose_bbox('yolo')


if __name__ == '__main__':
    unittest.main()
