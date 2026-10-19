"""Canonical Cube8 geometry and ordered 2-D keypoint contract."""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Sequence

CANONICAL_VERTEX_LABELS = (
    "x_neg_y_neg_z_neg",
    "x_pos_y_neg_z_neg",
    "x_pos_y_pos_z_neg",
    "x_neg_y_pos_z_neg",
    "x_neg_y_neg_z_pos",
    "x_pos_y_neg_z_pos",
    "x_pos_y_pos_z_pos",
    "x_neg_y_pos_z_pos",
)
CUBE_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)
GEOMETRY_ID = "cube-axis-v1"


@dataclass(frozen=True)
class Vertex3D:
    label: str
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class PoseKeypoint2D:
    label: str
    x: float
    y: float
    confidence: float
    visibility: int


def cube_vertices(side_length_m: float) -> tuple[Vertex3D, ...]:
    side = float(side_length_m)
    if not math.isfinite(side) or side <= 0.0:
        raise ValueError("side_length_m must be finite and positive")
    h = side / 2.0
    coords = (
        (-h, -h, -h), (+h, -h, -h), (+h, +h, -h), (-h, +h, -h),
        (-h, -h, +h), (+h, -h, +h), (+h, +h, +h), (-h, +h, +h),
    )
    return tuple(Vertex3D(label, *xyz)
                 for label, xyz in zip(CANONICAL_VERTEX_LABELS, coords))


def ordered_keypoints(points: Sequence[Sequence[float]], scores: Sequence[float],
                      *, visible_confidence: float = 0.50) -> tuple[PoseKeypoint2D, ...]:
    if len(points) != 8 or len(scores) != 8:
        raise ValueError("Cube8 requires exactly 8 points and 8 scores")
    if not 0.0 <= float(visible_confidence) <= 1.0:
        raise ValueError("visible_confidence must be in [0,1]")
    out = []
    for label, point, raw_score in zip(
            CANONICAL_VERTEX_LABELS, points, scores):
        if len(point) < 2:
            raise ValueError(f"{label}: point must contain x,y")
        x, y = float(point[0]), float(point[1])
        score = max(0.0, min(1.0, float(raw_score)))
        if not all(math.isfinite(v) for v in (x, y, score)):
            raise ValueError(f"{label}: non-finite point")
        visibility = 0 if (
            x == 0.0 and y == 0.0 and score == 0.0) else (
            2 if score >= visible_confidence else 1)
        out.append(PoseKeypoint2D(label, x, y, score, visibility))
    return tuple(out)
