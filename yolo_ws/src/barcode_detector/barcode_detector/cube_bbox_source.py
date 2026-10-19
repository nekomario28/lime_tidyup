"""Where a cube bounding box comes from, with the model-free pose in mind.

The pose itself no longer needs a learned model (see
``doc/design/2026-10-20-model-free-pose-route.md``); what it still needs is a
bounding box. This module makes that a swappable source instead of a hard
dependency on YOLO:

``detector``
    Use the box a detector already produced. Same behaviour as before; the sim
    keeps working unchanged.

``depth_blob``
    Find the cube as the nearest compact blob inside a working-distance band.
    Needs a depth image, no training, a few milliseconds. This is the intended
    source on the real robot, where a plain table and a cluttered room make
    colour-only detection unusable (measured: the floor is the largest "white"
    region, 154 880 px against a 14 327 px face).

``colour_pair``
    Kept for single-colour backdrops only, and off by default. The same
    measurement showed it cannot be trusted on a light floor.

Every rule is fail-closed: a box that does not clearly describe one compact
object at a plausible distance is not returned, and the caller keeps searching.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: Bbox sources this module knows.
BOX_SOURCES = ('detector', 'depth_blob', 'colour_pair')


class BboxSourceError(ValueError):
    """Raised when a bbox source is misconfigured or its input is unusable."""


@dataclass(frozen=True)
class DepthBlobConfig:
    """Acceptance rules for a depth blob, all relative and in metres."""

    minimum_range_m: float = 0.12
    maximum_range_m: float = 0.90
    minimum_area_fraction: float = 0.004
    #: A blob bigger than this is a surface, not an object on a table.
    maximum_area_fraction: float = 0.35
    #: How many image edges the blob may touch; a cube sits inside the frame.
    maximum_border_contacts: int = 1
    minimum_compactness: float = 0.35
    minimum_aspect: float = 0.45
    maximum_aspect: float = 2.20
    maximum_depth_spread_m: float = 0.12
    centre_band_fraction: float = 0.75
    close_kernel_px: int = 5
    margin_fraction: float = 0.12

    def validate(self) -> None:
        """Reject a configuration that cannot accept anything."""
        values = (
            self.minimum_range_m,
            self.maximum_range_m,
            self.minimum_area_fraction,
            self.maximum_area_fraction,
            self.minimum_compactness,
            self.minimum_aspect,
            self.maximum_aspect,
            self.maximum_depth_spread_m,
            self.centre_band_fraction,
            self.margin_fraction,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise BboxSourceError('depth blob configuration is not finite')
        if not 0.0 < self.minimum_range_m < self.maximum_range_m:
            raise BboxSourceError('depth range band is invalid')
        if not 0.0 < self.minimum_area_fraction < self.maximum_area_fraction:
            raise BboxSourceError(
                'area fraction band must be in (0, maximum_area_fraction)'
            )
        if not self.minimum_area_fraction < self.maximum_area_fraction <= 1.0:
            raise BboxSourceError('maximum area fraction must be in (0,1]')
        if self.maximum_border_contacts < 0:
            raise BboxSourceError('border contact limit must not be negative')
        if not 0.0 < self.minimum_compactness <= 1.0:
            raise BboxSourceError('minimum compactness must be in (0,1]')
        if not 0.0 < self.minimum_aspect <= self.maximum_aspect:
            raise BboxSourceError('aspect band is invalid')
        if self.maximum_depth_spread_m <= 0.0:
            raise BboxSourceError('depth spread must be positive')


@dataclass(frozen=True)
class BboxResult:
    """One accepted box and the evidence that produced it."""

    box_xyxy: tuple[float, float, float, float]
    source: str
    quality: float
    detail: dict


def depth_blob_bbox(
    depth_image,
    *,
    config=None,
    roi_xyxy=None,
):
    """Return the nearest plausible object box from a depth image.

    ``depth_image`` is HxW in metres (NaN, inf or 0 for missing). The blob must
    sit inside the range band, be compact enough to be an object rather than a
    wall, keep a sane aspect ratio, and not be a sloped surface. The nearest
    surviving blob wins.
    """
    import cv2

    box_config = config or DepthBlobConfig()
    box_config.validate()
    depth = np.asarray(depth_image, dtype=np.float32)
    if depth.ndim != 2 or depth.size == 0:
        raise BboxSourceError('depth image must be a non-empty 2-D array')

    height, width = depth.shape
    if roi_xyxy is None:
        x0, y0, x1, y1 = 0, 0, width, height
    else:
        x0, y0, x1, y1 = (int(round(value)) for value in roi_xyxy)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(width, x1), min(height, y1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        raise BboxSourceError('depth ROI is too small')

    window = depth[y0:y1, x0:x1]
    finite = np.isfinite(window)
    band = (
        finite
        & (window >= box_config.minimum_range_m)
        & (window <= box_config.maximum_range_m)
    )
    mask = band.astype(np.uint8) * 255
    kernel = np.ones(
        (max(1, int(box_config.close_kernel_px)),
         max(1, int(box_config.close_kernel_px))),
        np.uint8,
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    if count < 2:
        raise BboxSourceError('no depth blob inside the working band')

    roi_area = float((x1 - x0) * (y1 - y0))
    candidates = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < box_config.minimum_area_fraction * roi_area:
            continue
        if area > box_config.maximum_area_fraction * roi_area:
            # a surface filling the frame is a wall or a table, not the cube
            continue
        left = int(stats[index, cv2.CC_STAT_LEFT])
        top = int(stats[index, cv2.CC_STAT_TOP])
        box_width = int(stats[index, cv2.CC_STAT_WIDTH])
        box_height = int(stats[index, cv2.CC_STAT_HEIGHT])
        if box_width <= 0 or box_height <= 0:
            continue
        border_contacts = 0
        if left <= 2:
            border_contacts += 1
        if top <= 2:
            border_contacts += 1
        if left + box_width >= (x1 - x0) - 2:
            border_contacts += 1
        if top + box_height >= (y1 - y0) - 2:
            border_contacts += 1
        if border_contacts > box_config.maximum_border_contacts:
            # the cube is a blob inside the view, not something clipped by it
            continue
        aspect = box_width / float(box_height)
        if not (box_config.minimum_aspect <= aspect
                <= box_config.maximum_aspect):
            continue
        compactness = area / float(box_width * box_height)
        if compactness < box_config.minimum_compactness:
            continue
        centre_x = left + box_width / 2.0
        band_width = box_config.centre_band_fraction * (x1 - x0)
        if abs(centre_x - (x1 - x0) / 2.0) > band_width / 2.0:
            continue
        values = window[labels == index]
        values = values[np.isfinite(values)]
        if values.size < 16:
            continue
        median = float(np.median(values))
        spread = float(np.std(values))
        if spread > box_config.maximum_depth_spread_m:
            continue
        # depth is monotone with distance, so the nearest blob is the smallest
        # median; the cube is what the robot is looking at, not the wall behind.
        candidates.append({
            'index': index,
            'area': area,
            'median_depth_m': median,
            'depth_spread_m': spread,
            'compactness': compactness,
            'aspect': aspect,
            'box': (left + x0, top + y0,
                    left + x0 + box_width, top + y0 + box_height),
        })

    if not candidates:
        raise BboxSourceError('no depth blob passed the acceptance rules')

    near_wall = min(candidates, key=lambda item: item['median_depth_m'])
    margin = box_config.margin_fraction * max(
        near_wall['box'][2] - near_wall['box'][0],
        near_wall['box'][3] - near_wall['box'][1],
    )
    box = (
        max(0.0, near_wall['box'][0] - margin),
        max(0.0, near_wall['box'][1] - margin),
        min(float(width), near_wall['box'][2] + margin),
        min(float(height), near_wall['box'][3] + margin),
    )
    # quality: how well the blob satisfies the shape rules, still fail-closed
    quality = min(
        1.0,
        near_wall['compactness'],
        near_wall['area'] / (box_config.minimum_area_fraction * roi_area * 4.0),
    )
    return BboxResult(
        box_xyxy=box,
        source='depth_blob',
        quality=float(max(0.0, min(1.0, quality))),
        detail={
            'median_depth_m': round(near_wall['median_depth_m'], 4),
            'depth_spread_m': round(near_wall['depth_spread_m'], 4),
            'compactness': round(near_wall['compactness'], 3),
            'aspect': round(near_wall['aspect'], 3),
            'area_px': near_wall['area'],
            'candidates': len(candidates),
        },
    )


def detector_bbox(box_xyxy, *, quality=1.0):
    """Accept a detector box as-is, validating only its shape."""
    values = tuple(float(value) for value in box_xyxy)
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise BboxSourceError('detector box must be four finite numbers')
    left, top, right, bottom = values
    if right - left < 1.0 or bottom - top < 1.0:
        raise BboxSourceError('detector box is degenerate')
    return BboxResult(
        box_xyxy=(left, top, right, bottom),
        source='detector',
        quality=float(quality),
        detail={},
    )


def choose_bbox(
    source,
    *,
    detector_box=None,
    detector_quality=1.0,
    depth_image=None,
    depth_config=None,
    roi_xyxy=None,
):
    """Dispatch to one bbox source, keeping the fail-closed behaviour."""
    if source not in BOX_SOURCES:
        raise BboxSourceError(f'unknown bbox source: {source!r}')
    if source == 'detector':
        if detector_box is None:
            raise BboxSourceError('detector box required for source "detector"')
        return detector_bbox(detector_box, quality=detector_quality)
    if source == 'depth_blob':
        if depth_image is None:
            raise BboxSourceError('depth image required for source "depth_blob"')
        return depth_blob_bbox(
            depth_image, config=depth_config, roi_xyxy=roi_xyxy
        )
    raise BboxSourceError(
        'source "colour_pair" is an experiment, not a production source; '
        'measured unusable on a light floor'
    )
