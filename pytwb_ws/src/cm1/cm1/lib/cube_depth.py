"""Robust depth estimates for a detected cube bounding box."""

from __future__ import annotations

from collections import deque
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


@dataclass(frozen=True)
class CubeGeometryCheck:
    """Known-size projection evidence for one detected bounding box."""

    accepted: bool
    expected_pixel_size: float
    observed_pixel_size: float
    minimum_pixel_size: float
    maximum_pixel_size: float


@dataclass(frozen=True)
class TemporalConfirmation:
    """Current K-of-N confirmation state for one nearby candidate track."""

    confirmed: bool
    valid_count: int
    window_size: int
    observation_count: int
    track_reset: bool


class CubeTemporalGate:
    """Confirm nearby candidates with a small K-of-N boolean history."""

    def __init__(
        self,
        *,
        window_size=5,
        required_count=3,
        maximum_center_distance_px=60.0,
    ):
        """Configure history length, quorum and center association limit."""
        window_size = int(window_size)
        required_count = int(required_count)
        maximum_center_distance_px = float(maximum_center_distance_px)
        if window_size < 1:
            raise CubeDepthError('temporal window size must be positive')
        if required_count < 1 or required_count > window_size:
            raise CubeDepthError(
                'temporal required count must be within the window'
            )
        if (
            not math.isfinite(maximum_center_distance_px)
            or maximum_center_distance_px < 0.0
        ):
            raise CubeDepthError(
                'temporal center distance must be finite and non-negative'
            )

        self.window_size = window_size
        self.required_count = required_count
        self.maximum_center_distance_px = maximum_center_distance_px
        self._history = deque(maxlen=window_size)
        self._last_center = None

    def update(self, center, *, valid):
        """Add one observation and return its confirmation state."""
        track_reset = False
        parsed_center = _finite_center(center)
        if parsed_center is not None:
            if self._last_center is not None:
                distance_px = math.hypot(
                    parsed_center[0] - self._last_center[0],
                    parsed_center[1] - self._last_center[1],
                )
                if distance_px > self.maximum_center_distance_px:
                    self._history.clear()
                    track_reset = True
            self._last_center = parsed_center

        self._history.append(bool(valid) and parsed_center is not None)
        valid_count = sum(self._history)
        return TemporalConfirmation(
            confirmed=valid_count >= self.required_count,
            valid_count=valid_count,
            window_size=self.window_size,
            observation_count=len(self._history),
            track_reset=track_reset,
        )


def check_known_size_geometry(
    box,
    *,
    distance_m,
    focal_length_px,
    side_length_m=0.057,
    minimum_scale=0.60,
    maximum_scale=1.90,
):
    """Compare bbox extent with a loose known-size pinhole projection."""
    try:
        x_min, y_min, x_max, y_max = (float(value) for value in box)
        distance_m = float(distance_m)
        focal_length_px = float(focal_length_px)
        side_length_m = float(side_length_m)
        minimum_scale = float(minimum_scale)
        maximum_scale = float(maximum_scale)
    except (TypeError, ValueError):
        raise CubeDepthError('geometry inputs must be numeric')

    values = (
        x_min,
        y_min,
        x_max,
        y_max,
        distance_m,
        focal_length_px,
        side_length_m,
        minimum_scale,
        maximum_scale,
    )
    if not all(math.isfinite(value) for value in values):
        raise CubeDepthError('geometry inputs must be finite')
    if x_max <= x_min or y_max <= y_min:
        raise CubeDepthError('box has no positive area')
    if distance_m <= 0.0 or focal_length_px <= 0.0:
        raise CubeDepthError('distance and focal length must be positive')
    if side_length_m <= 0.0:
        raise CubeDepthError('cube side length must be positive')
    if minimum_scale <= 0.0 or maximum_scale < minimum_scale:
        raise CubeDepthError('geometry scale range is invalid')

    expected_pixel_size = focal_length_px * side_length_m / distance_m
    observed_pixel_size = max(x_max - x_min, y_max - y_min)
    minimum_pixel_size = expected_pixel_size * minimum_scale
    maximum_pixel_size = expected_pixel_size * maximum_scale
    return CubeGeometryCheck(
        accepted=(
            minimum_pixel_size
            <= observed_pixel_size
            <= maximum_pixel_size
        ),
        expected_pixel_size=expected_pixel_size,
        observed_pixel_size=observed_pixel_size,
        minimum_pixel_size=minimum_pixel_size,
        maximum_pixel_size=maximum_pixel_size,
    )


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


def _finite_center(center):
    if center is None:
        return None
    try:
        center_x, center_y = (float(value) for value in center)
    except (TypeError, ValueError):
        raise CubeDepthError('candidate center must contain two numbers')
    if not math.isfinite(center_x) or not math.isfinite(center_y):
        raise CubeDepthError('candidate center must be finite')
    return (center_x, center_y)
