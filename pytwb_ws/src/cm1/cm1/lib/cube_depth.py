"""Robust depth estimates for a detected cube bounding box."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


class CubeDepthError(ValueError):
    """Raised when depth evidence is unusable or inconsistent."""


@dataclass(frozen=True)
class CubeDepthEstimate:
    """Accepted depth and the quality evidence supporting it."""

    distance_m: float
    valid_ratio: float
    mad_m: float
    sample_count: int
    roi_xyxy: tuple[int, int, int, int]


def estimate_cube_depth(
    depth_image,
    box,
    *,
    encoding='',
    rgb_width=848.0,
    rgb_height=480.0,
    roi_fraction=0.0,
    center_radius_px=5,
    minimum_valid_ratio=0.0,
    maximum_mad_m=10.0,
    minimum_distance_m=0.05,
    maximum_distance_m=10.0,
):
    """Return a robust cube distance from the inner detection region."""
    image = np.asarray(depth_image)
    if image.ndim < 2 or image.size == 0:
        raise CubeDepthError('depth image is empty')

    try:
        x_min, y_min, x_max, y_max = (
            float(value) for value in box
        )
    except (TypeError, ValueError):
        raise CubeDepthError('box must contain four numeric values')

    values = (x_min, y_min, x_max, y_max, rgb_width, rgb_height)
    if not all(math.isfinite(float(value)) for value in values):
        raise CubeDepthError('box or RGB dimensions contain non-finite values')
    if x_max <= x_min or y_max <= y_min:
        raise CubeDepthError('box has no positive area')
    if float(rgb_width) <= 0.0 or float(rgb_height) <= 0.0:
        raise CubeDepthError('RGB dimensions must be positive')

    depth_height, depth_width = image.shape[:2]
    scale_x = depth_width / float(rgb_width)
    scale_y = depth_height / float(rgb_height)
    scaled_box = (
        x_min * scale_x,
        y_min * scale_y,
        x_max * scale_x,
        y_max * scale_y,
    )
    roi = _depth_roi(
        scaled_box,
        depth_width,
        depth_height,
        float(roi_fraction),
        int(center_radius_px),
    )
    roi_x1, roi_y1, roi_x2, roi_y2 = roi
    region = image[roi_y1:roi_y2, roi_x1:roi_x2].astype(np.float64)
    if region.size == 0:
        raise CubeDepthError('depth ROI is empty')

    finite_positive = region[np.isfinite(region) & (region > 0.0)]
    if finite_positive.size == 0:
        raise CubeDepthError('depth ROI has no finite positive samples')

    distances_m = _to_metres(finite_positive, encoding)
    in_range = distances_m[
        (distances_m >= float(minimum_distance_m))
        & (distances_m <= float(maximum_distance_m))
    ]
    valid_ratio = float(in_range.size) / float(region.size)
    if in_range.size == 0:
        raise CubeDepthError('depth ROI has no samples in the valid range')
    if valid_ratio < float(minimum_valid_ratio):
        raise CubeDepthError(
            f'depth valid ratio {valid_ratio:.3f} is below '
            f'{float(minimum_valid_ratio):.3f}'
        )

    distance_m = float(np.median(in_range))
    mad_m = float(np.median(np.abs(in_range - distance_m)))
    if not math.isfinite(distance_m) or not math.isfinite(mad_m):
        raise CubeDepthError('depth estimate is non-finite')
    if mad_m > float(maximum_mad_m):
        raise CubeDepthError(
            f'depth MAD {mad_m:.3f}m exceeds '
            f'{float(maximum_mad_m):.3f}m'
        )

    return CubeDepthEstimate(
        distance_m=distance_m,
        valid_ratio=valid_ratio,
        mad_m=mad_m,
        sample_count=int(in_range.size),
        roi_xyxy=roi,
    )


def consistent_depth_median(distances_m, *, maximum_spread_m):
    """Return the median only when repeated depth estimates agree."""
    try:
        values = tuple(float(value) for value in distances_m)
    except (TypeError, ValueError):
        raise CubeDepthError('depth samples must be numeric')
    if not values:
        raise CubeDepthError('at least one depth sample is required')
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise CubeDepthError('depth samples must be finite and positive')

    spread_m = max(values) - min(values)
    if spread_m > float(maximum_spread_m):
        raise CubeDepthError(
            f'depth spread {spread_m:.3f}m exceeds '
            f'{float(maximum_spread_m):.3f}m'
        )
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _depth_roi(
    scaled_box,
    depth_width,
    depth_height,
    roi_fraction,
    center_radius_px,
):
    x_min, y_min, x_max, y_max = scaled_box
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0

    if roi_fraction <= 0.0:
        radius = max(0, center_radius_px)
        center_x_int = max(0, min(int(center_x), depth_width - 1))
        center_y_int = max(0, min(int(center_y), depth_height - 1))
        x1 = max(0, center_x_int - radius)
        y1 = max(0, center_y_int - radius)
        x2 = min(depth_width, center_x_int + radius + 1)
        y2 = min(depth_height, center_y_int + radius + 1)
        return (x1, y1, x2, y2)

    if roi_fraction > 1.0:
        raise CubeDepthError('ROI fraction must be in (0, 1]')
    half_width = max(0.5, (x_max - x_min) * roi_fraction / 2.0)
    half_height = max(0.5, (y_max - y_min) * roi_fraction / 2.0)

    x1 = max(0, min(int(math.floor(center_x - half_width)), depth_width - 1))
    y1 = max(0, min(int(math.floor(center_y - half_height)), depth_height - 1))
    x2 = max(x1 + 1, min(int(math.ceil(center_x + half_width)), depth_width))
    y2 = max(y1 + 1, min(int(math.ceil(center_y + half_height)), depth_height))
    return (x1, y1, x2, y2)


def _to_metres(raw_values, encoding):
    if encoding in ('16UC1', 'mono16'):
        return raw_values / 1000.0
    if encoding == '32FC1':
        return raw_values
    if float(np.median(raw_values)) > 20.0:
        return raw_values / 1000.0
    return raw_values
