"""Adaptive two-face color-region front-end for view-local Cube8 geometry.

Only the Y-axis opposite color pair (blue/green) and Z-axis pair (white/yellow)
are selected. Red/orange X-face identity is intentionally not estimated here.
The returned geometry keeps local 2..7 as observed, local 1 projectively derived,
and local 0 hidden/unobserved.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from .cube8_two_face_view_local import (
    ViewLocalTwoFaceGeometry,
    recover_view_local_from_yz_quads,
)


class Cube8TwoFaceColorError(ValueError):
    pass


@dataclass(frozen=True)
class TwoFaceColorFrontendConfig:
    search_bbox_expansion: float = 1.50
    prototype_interior_shrink_fraction: float = 0.08
    seed_min_saturation: int = 60
    seed_min_value: int = 30
    white_max_saturation: int = 70
    white_min_value: int = 120
    maximum_seed_hue_distance: float = 20.0
    minimum_axis_seed_fraction: float = 0.003
    maximum_face_to_background_distance_ratio: float = 0.85
    morphology_close_over_bbox_diag: float = 0.012
    minimum_component_area_over_bbox_area: float = 0.005
    polygon_epsilon_min_fraction: float = 0.004
    polygon_epsilon_max_fraction: float = 0.120
    polygon_epsilon_steps: int = 80
    silhouette_refinement_enabled: bool = True
    silhouette_background_distance: float = 30.0
    silhouette_epsilon_fraction: float = 0.02
    silhouette_snap_max_fraction: float = 0.09
    silhouette_border_ring_px: int = 3

    def validate(self) -> None:
        floats = (
            self.search_bbox_expansion,
            self.prototype_interior_shrink_fraction,
            self.maximum_seed_hue_distance,
            self.minimum_axis_seed_fraction,
            self.maximum_face_to_background_distance_ratio,
            self.morphology_close_over_bbox_diag,
            self.minimum_component_area_over_bbox_area,
            self.polygon_epsilon_min_fraction,
            self.polygon_epsilon_max_fraction,
            self.silhouette_background_distance,
            self.silhouette_epsilon_fraction,
            self.silhouette_snap_max_fraction,
        )
        if not all(math.isfinite(float(v)) for v in floats):
            raise Cube8TwoFaceColorError(
                "two-face color configuration contains non-finite values")
        if self.search_bbox_expansion < 1.0:
            raise Cube8TwoFaceColorError("search_bbox_expansion must be >= 1")
        if not 0.0 <= self.prototype_interior_shrink_fraction < 0.45:
            raise Cube8TwoFaceColorError(
                "prototype interior shrink is invalid")
        if not 0.0 < self.minimum_axis_seed_fraction < 1.0:
            raise Cube8TwoFaceColorError(
                "minimum_axis_seed_fraction must be in (0,1)")
        if not 0.0 < self.maximum_face_to_background_distance_ratio < 2.0:
            raise Cube8TwoFaceColorError(
                "face/background distance ratio is invalid")
        if self.polygon_epsilon_max_fraction <= self.polygon_epsilon_min_fraction:
            raise Cube8TwoFaceColorError("polygon epsilon range is invalid")
        if self.silhouette_background_distance <= 0.0:
            raise Cube8TwoFaceColorError(
                "silhouette background distance must be positive"
            )
        if not 0.0 < self.silhouette_epsilon_fraction < 0.2:
            raise Cube8TwoFaceColorError(
                "silhouette epsilon fraction must be in (0,0.2)"
            )
        if not 0.0 < self.silhouette_snap_max_fraction < 0.5:
            raise Cube8TwoFaceColorError(
                "silhouette snap limit must be in (0,0.5)"
            )
        if self.silhouette_border_ring_px < 1:
            raise Cube8TwoFaceColorError(
                "silhouette border ring must be at least one pixel"
            )
        if self.polygon_epsilon_steps < 2:
            raise Cube8TwoFaceColorError("polygon_epsilon_steps must be >= 2")


@dataclass(frozen=True)
class FaceQuadrilateral:
    color: str
    vertices: tuple[tuple[float, float], ...]
    component_area_px: int
    mask_support_px: int


@dataclass(frozen=True)
class TwoFaceColorObservation:
    visible_y_color: str
    visible_z_color: str
    support_pixels: Mapping[str, int]
    lab_prototypes: Mapping[str, tuple[float, float, float]]
    background_prototype: tuple[float, float, float]
    search_box_xyxy: tuple[int, int, int, int]
    face_quads: Mapping[str, FaceQuadrilateral]
    geometry: ViewLocalTwoFaceGeometry

    @property
    def x_color_semantics_required(self) -> bool:
        return False


#: OpenCV hue centres for the chromatic cube colours. Red wraps through 0, which
#: the seed mask already handles with a circular distance.
_HUE_CENTER = {
    "red": 3.0,
    "orange": 16.0,
    "yellow": 30.0,
    "green": 60.0,
    "blue": 110.0,
}
#: Colour vocabulary the frontend can name; order only breaks support ties.
_FACE_COLORS = ("white", "red", "orange", "yellow", "green", "blue")


def recover_two_face_view_local_from_bgr(
    frame,
    detector_bbox_xyxy: Sequence[float],
    *,
    config: TwoFaceColorFrontendConfig = TwoFaceColorFrontendConfig(),
) -> TwoFaceColorObservation:
    import cv2
    import numpy as np

    config.validate()
    if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
        raise Cube8TwoFaceColorError("frame must be a decoded image")
    height, width = int(frame.shape[0]), int(frame.shape[1])
    if height <= 0 or width <= 0:
        raise Cube8TwoFaceColorError("frame is empty")
    bbox = _validate_bbox(detector_bbox_xyxy, width, height)
    diagonal = math.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1])
    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    if diagonal <= 0.0 or bbox_area <= 0.0:
        raise Cube8TwoFaceColorError("detector bbox is degenerate")

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    selected, support, prototypes, background, search = _bootstrap_yz_model(
        hsv, lab, bbox, config)
    masks = {
        color: _high_confidence_face_mask(
            hsv, lab, color, prototypes, background, search, diagonal, config
        )
        for color in selected.values()
    }
    minimum_area = max(
        8, int(
            round(
                config.minimum_component_area_over_bbox_area * bbox_area)))
    quads = {
        color: _fit_face_quad(mask, color, search, minimum_area, config)
        for color, mask in masks.items()
    }
    y_color, z_color = selected["y"], selected["z"]
    if config.silhouette_refinement_enabled:
        snapped = snap_quads_to_silhouette(
            frame,
            {
                "search_box": search,
                "vertices": {
                    color: quad.vertices for color, quad in quads.items()
                },
            },
            config,
            diagonal,
        )
        if snapped is not None:
            quads = {
                color: FaceQuadrilateral(
                    color=quad.color,
                    vertices=snapped[color],
                    component_area_px=quad.component_area_px,
                    mask_support_px=quad.mask_support_px,
                )
                for color, quad in quads.items()
            }
    geometry = recover_view_local_from_yz_quads(
        quads[y_color].vertices,
        quads[z_color].vertices,
        bbox,
    )
    return TwoFaceColorObservation(
        visible_y_color=y_color,
        visible_z_color=z_color,
        support_pixels=support,
        lab_prototypes=prototypes,
        background_prototype=background,
        search_box_xyxy=search,
        face_quads=quads,
        geometry=geometry,
    )


def _bootstrap_yz_model(hsv, lab, bbox, config):
    import numpy as np

    height, width = int(hsv.shape[0]), int(hsv.shape[1])
    x0, y0, x1, y1 = bbox
    margin_x = (x1 - x0) * config.prototype_interior_shrink_fraction
    margin_y = (y1 - y0) * config.prototype_interior_shrink_fraction
    ix0 = max(0, int(math.floor(x0 + margin_x)))
    iy0 = max(0, int(math.floor(y0 + margin_y)))
    ix1 = min(width, int(math.ceil(x1 - margin_x)))
    iy1 = min(height, int(math.ceil(y1 - margin_y)))
    if ix1 <= ix0 or iy1 <= iy0:
        raise Cube8TwoFaceColorError("prototype interior is empty")
    roi = hsv[iy0:iy1, ix0:ix1]
    seed = {color: _seed_mask(roi, color, config) for color in _FACE_COLORS}
    support = {color: int(np.count_nonzero(mask))
               for color, mask in seed.items()}
    # The two faces that are actually visible are the two largest colour regions
    # inside the detector box; a third visible face is simply not used. Picking
    # by support instead of a fixed colour pair is what lets any two of the six
    # standard colours form the view.
    ranked = sorted(
        _FACE_COLORS,
        key=lambda name: (-support[name], name),
    )
    selected = {"y": ranked[0], "z": ranked[1]}
    minimum_support = max(
        1, int(
            math.ceil(
                config.minimum_axis_seed_fraction * roi.shape[0] * roi.shape[1])))
    for axis, color in selected.items():
        if support[color] < minimum_support:
            raise Cube8TwoFaceColorError(
                f"insufficient {axis}-axis face-color seed support: "
                f"{color}={support[color]} < {minimum_support}"
            )
    roi_lab = lab[iy0:iy1, ix0:ix1]
    prototypes = {}
    for color in selected.values():
        values = roi_lab[seed[color]]
        if values.size == 0:
            raise Cube8TwoFaceColorError(f"empty Lab prototype for {color}")
        prototype = np.median(values, axis=0)
        prototypes[color] = tuple(float(v) for v in prototype)

    search = _expanded_box(bbox, config.search_bbox_expansion, width, height)
    sx0, sy0, sx1, sy1 = search
    ring = np.zeros((height, width), dtype=bool)
    ring[sy0:sy1, sx0:sx1] = True
    bx0 = max(0, int(math.floor(x0)))
    by0 = max(0, int(math.floor(y0)))
    bx1 = min(width, int(math.ceil(x1)))
    by1 = min(height, int(math.ceil(y1)))
    ring[by0:by1, bx0:bx1] = False
    values = lab[ring]
    if values.size == 0:
        raise Cube8TwoFaceColorError("background ring contains no pixels")
    background = tuple(float(v) for v in np.median(values, axis=0))
    return selected, support, prototypes, background, search


def _high_confidence_face_mask(
        hsv,
        lab,
        color,
        prototypes,
        background,
        search,
        diagonal,
        config):
    import cv2
    import numpy as np

    sx0, sy0, sx1, sy1 = search
    roi_hsv = hsv[sy0:sy1, sx0:sx1]
    roi_lab = lab[sy0:sy1, sx0:sx1]
    if roi_hsv.size == 0:
        raise Cube8TwoFaceColorError("search ROI is empty")
    seed = _seed_mask(roi_hsv, color, config)
    face = np.asarray(prototypes[color], dtype=np.float32)
    bg = np.asarray(background, dtype=np.float32)
    face_distance = np.linalg.norm(roi_lab - face, axis=2)
    background_distance = np.linalg.norm(roi_lab - bg, axis=2)
    nearest_face = np.ones(face_distance.shape, dtype=bool)
    for other, prototype in prototypes.items():
        if other != color:
            nearest_face &= face_distance <= np.linalg.norm(
                roi_lab - np.asarray(prototype, dtype=np.float32), axis=2
            )
    mask = (
        seed
        & nearest_face
        & (
            face_distance
            <= config.maximum_face_to_background_distance_ratio
            * np.maximum(background_distance, 1e-6)
        )
    ).astype(np.uint8) * 255
    kernel_size = max(
        3, int(
            round(
                config.morphology_close_over_bbox_diag * diagonal)))
    kernel_size += int(kernel_size % 2 == 0)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones(
        (kernel_size, kernel_size), dtype=np.uint8), iterations=1)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, np.ones(
            (3, 3), dtype=np.uint8), iterations=1)
    return mask


def _seed_mask(hsv_roi, color, config):
    import numpy as np
    h = hsv_roi[:, :, 0].astype(np.float32)
    s = hsv_roi[:, :, 1]
    v = hsv_roi[:, :, 2]
    if color == "white":
        return (
            s <= config.white_max_saturation) & (
            v >= config.white_min_value)
    center = _HUE_CENTER.get(color)
    if center is None:
        raise Cube8TwoFaceColorError(
            f"unsupported two-face seed color: {color}")
    distance = np.minimum(np.abs(h - center), 180.0 - np.abs(h - center))
    return (
        (s >= config.seed_min_saturation)
        & (v >= config.seed_min_value)
        & (distance <= config.maximum_seed_hue_distance)
    )


def _fit_face_quad(mask, color, search_box, minimum_area, config):
    import cv2
    import numpy as np

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8)
    candidates = [
        i for i in range(1, count)
        if int(stats[i, cv2.CC_STAT_AREA]) >= minimum_area
    ]
    if not candidates:
        raise Cube8TwoFaceColorError(
            f"no sufficiently large region for {color}")
    component = max(candidates, key=lambda i: int(stats[i, cv2.CC_STAT_AREA]))
    yy, xx = np.where(labels == component)
    if len(xx) < 4:
        raise Cube8TwoFaceColorError(
            f"insufficient component points for {color}")
    hull = cv2.convexHull(np.column_stack(
        (xx, yy)).astype(np.int32)).reshape(-1, 2)
    if len(hull) < 4:
        raise Cube8TwoFaceColorError(
            f"convex hull has fewer than four vertices for {color}")
    contour = hull.reshape(-1, 1, 2)
    perimeter = float(cv2.arcLength(contour, True))
    polygon = None
    for step in range(config.polygon_epsilon_steps):
        alpha = step / (config.polygon_epsilon_steps - 1)
        fraction = (
            config.polygon_epsilon_min_fraction * (1.0 - alpha)
            + config.polygon_epsilon_max_fraction * alpha
        )
        approximation = cv2.approxPolyDP(
            contour, fraction * perimeter, True).reshape(-1, 2)
        if len(approximation) == 4:
            polygon = approximation
            break
        if len(approximation) < 4:
            break
    if polygon is None:
        raise Cube8TwoFaceColorError(
            f"cannot approximate {color} face as a quadrilateral")
    sx0, sy0, _, _ = search_box
    vertices = tuple((float(x + sx0), float(y + sy0)) for x, y in polygon)
    return FaceQuadrilateral(
        color=color,
        vertices=vertices,
        component_area_px=int(stats[component, cv2.CC_STAT_AREA]),
        mask_support_px=int(np.count_nonzero(mask)),
    )


def silhouette_corners(frame, search_box, config):
    """Return the outer corners of the cube silhouette inside ``search_box``.

    The colour-region quads are inset from the cube by the sticker gaps and the
    core rim, which was measured at 4-17 px against a known pose; the silhouette
    is the object outline itself, measured at 0.5-2.8 px on the same frame. The
    background prototype comes from the search-box border ring, so no absolute
    background colour is assumed.
    """
    import cv2
    import numpy as np

    x0, y0, x1, y1 = (int(round(value)) for value in search_box)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(frame.shape[1], x1), min(frame.shape[0], y1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        raise Cube8TwoFaceColorError("silhouette search box is too small")
    crop = frame[y0:y1, x0:x1]
    ring = max(1, int(config.silhouette_border_ring_px))
    border = np.concatenate((
        crop[:ring].reshape(-1, 3),
        crop[-ring:].reshape(-1, 3),
        crop[:, :ring].reshape(-1, 3),
        crop[:, -ring:].reshape(-1, 3),
    ))
    background = np.median(border, axis=0)
    distance = np.linalg.norm(
        crop.astype(np.float32) - background, axis=2
    )
    foreground = (distance > config.silhouette_background_distance)
    mask = (foreground.astype(np.uint8)) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise Cube8TwoFaceColorError("silhouette produced no contour")
    contour = max(contours, key=cv2.contourArea)
    perimeter = float(cv2.arcLength(contour, True))
    if perimeter <= 0.0:
        raise Cube8TwoFaceColorError("silhouette contour is degenerate")
    polygon = cv2.approxPolyDP(
        contour, config.silhouette_epsilon_fraction * perimeter, True
    ).reshape(-1, 2)
    if len(polygon) < 4:
        raise Cube8TwoFaceColorError(
            "silhouette does not approximate to a polygon"
        )
    offset = np.array([x0, y0], dtype=np.float64)
    return tuple(
        (float(point[0] + offset[0]), float(point[1] + offset[1]))
        for point in polygon.astype(np.float64)
    )


def snap_quads_to_silhouette(corners, quads, config, diagonal):
    """Snap each quad corner to the nearest silhouette corner within a limit."""
    import math as _math

    try:
        candidates = silhouette_corners(corners, quads['search_box'], config)
    except Cube8TwoFaceColorError:
        return None
    limit = config.silhouette_snap_max_fraction * diagonal
    snapped = {}
    for color, vertices in quads['vertices'].items():
        updated = []
        for vertex in vertices:
            best = None
            best_distance = limit
            for candidate in candidates:
                distance = _math.hypot(
                    vertex[0] - candidate[0], vertex[1] - candidate[1]
                )
                if distance <= best_distance:
                    best = candidate
                    best_distance = distance
            # Snapping is independent per corner on purpose: the two faces
            # share an edge, so their shared corners must land on the same
            # silhouette corner rather than being forced onto distinct ones.
            updated.append(vertex if best is None else best)
        snapped[color] = tuple(updated)
    return snapped


def _expanded_box(bbox, factor, width, height):
    x0, y0, x1, y1 = bbox
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    half_w = 0.5 * (x1 - x0) * factor
    half_h = 0.5 * (y1 - y0) * factor
    return (
        max(0, int(math.floor(cx - half_w))),
        max(0, int(math.floor(cy - half_h))),
        min(width, int(math.ceil(cx + half_w))),
        min(height, int(math.ceil(cy + half_h))),
    )


def _validate_bbox(values, width, height):
    if len(values) != 4:
        raise Cube8TwoFaceColorError("detector bbox must contain four values")
    x0, y0, x1, y1 = (float(v) for v in values)
    if not all(math.isfinite(v) for v in (x0, y0, x1, y1)):
        raise Cube8TwoFaceColorError(
            "detector bbox contains non-finite values")
    if x1 <= x0 or y1 <= y0:
        raise Cube8TwoFaceColorError("detector bbox must have positive size")
    if x1 < 0.0 or y1 < 0.0 or x0 >= width or y0 >= height:
        raise Cube8TwoFaceColorError("detector bbox does not overlap image")
    return (x0, y0, x1, y1)
