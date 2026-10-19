"""Two visible face quads -> view-local Cube8 projective geometry.

Only the Y/Z face quads are treated as image observations. Local vertex 1 is
derived projectively; local vertex 0 is fully hidden. PnP observations therefore
contain only local vertices 2..7.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Sequence


class Cube8TwoFaceGeometryError(ValueError):
    pass


@dataclass(frozen=True)
class TwoFaceGeometryConfig:
    shared_edge_max_score_over_bbox_diag: float = 0.08
    minimum_observed_edge_length_over_bbox_diag: float = 0.08
    minimum_vertex_separation_over_bbox_diag: float = 0.08
    maximum_vertex_outside_bbox_margin_over_diag: float = 0.25
    require_zero_visible_edge_crossings: bool = True

    def validate(self) -> None:
        values = (
            self.shared_edge_max_score_over_bbox_diag,
            self.minimum_observed_edge_length_over_bbox_diag,
            self.minimum_vertex_separation_over_bbox_diag,
            self.maximum_vertex_outside_bbox_margin_over_diag,
        )
        if not all(math.isfinite(float(v)) and float(v)
                   >= 0.0 for v in values):
            raise Cube8TwoFaceGeometryError(
                "geometry gate values must be finite and non-negative")


@dataclass(frozen=True)
class ViewLocalTwoFaceGeometry:
    points: tuple[tuple[float, float] | None, ...]
    point_sources: tuple[str, ...]
    vanishing_x: tuple[float, float]
    vanishing_y: tuple[float, float]
    vanishing_z: tuple[float, float]
    shared_edge_score_over_bbox_diag: float
    bbox_diagonal_px: float
    gate_failures: tuple[str, ...]
    semantic_frame: str = "view_local_cube_symmetry"
    color_fixed_semantics_resolved: bool = False

    @property
    def gate_passed(self) -> bool:
        return not self.gate_failures

    @property
    def observed_indices(self) -> tuple[int, ...]:
        return (2, 3, 4, 5, 6, 7)

    @property
    def derived_indices(self) -> tuple[int, ...]:
        return (1,)

    def pnp_observations(self) -> dict[int, tuple[float, float]]:
        if not self.gate_passed:
            raise Cube8TwoFaceGeometryError(
                "two-face geometry gate failed: " +
                ",".join(
                    self.gate_failures))
        result = {}
        for index in self.observed_indices:
            point = self.points[index]
            if point is None:
                raise Cube8TwoFaceGeometryError(
                    f"observed local vertex {index} is missing")
            result[index] = point
        return result


VISIBLE_EDGES = (
    (1, 2), (2, 3),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (1, 5), (2, 6), (3, 7),
)
OBSERVED_EDGES = (
    (2, 3), (4, 5), (5, 6), (6, 7), (7, 4), (2, 6), (3, 7),
)


def recover_view_local_from_yz_quads(
    y_face_quad: Sequence[Sequence[float]],
    z_face_quad: Sequence[Sequence[float]],
    detector_bbox_xyxy: Sequence[float],
    *,
    config: TwoFaceGeometryConfig = TwoFaceGeometryConfig(),
) -> ViewLocalTwoFaceGeometry:
    """Recover local vertices from cyclic +Y/+Z face quads.

    Physical signs/colors are intentionally irrelevant. The caller may call whichever
    currently visible opposite-pair face is Y and Z in the view-local cube frame.
    """
    config.validate()
    bbox = _bbox(detector_bbox_xyxy)
    diag = math.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1])
    if diag <= 1e-9:
        raise Cube8TwoFaceGeometryError("detector bbox is degenerate")
    y = _quad(y_face_quad, "y_face_quad")
    z = _quad(z_face_quad, "z_face_quad")

    score, y0, y1, z0, z1, reverse = _match_shared_edge(y, z, diag)
    pairs = ((y0, z1), (y1, z0)) if reverse else ((y0, z0), (y1, z1))
    endpoints = tuple(_average(y[yi], z[zi]) for yi, zi in pairs)

    interior = tuple(_bbox_interiority(point, bbox) for point in endpoints)
    if abs(interior[0] - interior[1]) > 1e-9:
        f_slot = 0 if interior[0] > interior[1] else 1
    else:
        center = (0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3]))
        f_slot = 0 if _distance(
            endpoints[0],
            center) < _distance(
            endpoints[1],
            center) else 1
    g_slot = 1 - f_slot

    y_f, y_g = pairs[f_slot][0], pairs[g_slot][0]
    z_f, z_g = pairs[f_slot][1], pairs[g_slot][1]
    point_f, point_g = endpoints[f_slot], endpoints[g_slot]

    point_b = y[_other_neighbor(y_f, y_g)]
    point_c = y[_other_neighbor(y_g, y_f)]
    point_e = z[_other_neighbor(z_f, z_g)]
    point_d = z[_other_neighbor(z_g, z_f)]

    line_bf = _line(point_b, point_f)
    line_cg = _line(point_c, point_g)
    line_ef = _line(point_e, point_f)
    line_dg = _line(point_d, point_g)
    line_bc = _line(point_b, point_c)
    line_de = _line(point_d, point_e)
    line_fg = _line(point_f, point_g)
    vanishing_z = _least_squares_intersection((line_bf, line_cg))
    vanishing_y = _least_squares_intersection((line_ef, line_dg))
    vanishing_x = _least_squares_intersection((line_bc, line_de, line_fg))
    point_a = _least_squares_intersection(
        (_line(point_e, vanishing_z), _line(point_b, vanishing_y))
    )

    points = (
        None,
        point_a,
        point_b,
        point_c,
        point_d,
        point_e,
        point_f,
        point_g)
    sources = (
        "hidden_unobserved",
        "derived_projective_from_two_faces",
        "observed_y_face", "observed_y_face",
        "observed_z_face", "observed_z_face",
        "observed_shared_yz", "observed_shared_yz",
    )
    failures = []
    if score > config.shared_edge_max_score_over_bbox_diag:
        failures.append("shared_edge_mismatch")

    min_edge = config.minimum_observed_edge_length_over_bbox_diag * diag
    for first, second in OBSERVED_EDGES:
        if _distance(
            _required(
                points, first), _required(
                points, second)) < min_edge:
            failures.append("observed_edge_too_short")
            break

    min_sep = config.minimum_vertex_separation_over_bbox_diag * diag
    visible = [_required(points, index) for index in range(1, 8)]
    if min(
        _distance(first, second)
        for i, first in enumerate(visible)
        for second in visible[i + 1:]
    ) < min_sep:
        failures.append("visible_vertices_too_close")

    margin = config.maximum_vertex_outside_bbox_margin_over_diag * diag
    x1, y1b, x2, y2b = bbox
    if any(
        p[0] < x1 - margin or p[0] > x2 + margin or
        p[1] < y1b - margin or p[1] > y2b + margin
        for p in visible
    ):
        failures.append("visible_vertex_outside_bbox")

    if config.require_zero_visible_edge_crossings:
        segments = [(a, b, _required(points, a), _required(points, b))
                    for a, b in VISIBLE_EDGES]
        crossing = False
        for i, (a0, b0, p0, p1) in enumerate(segments):
            for a1, b1, q0, q1 in segments[i + 1:]:
                if {a0, b0} & {a1, b1}:
                    continue
                if _proper_cross(p0, p1, q0, q1):
                    crossing = True
                    break
            if crossing:
                break
        if crossing:
            failures.append("visible_cube_edges_cross")

    return ViewLocalTwoFaceGeometry(
        points=points,
        point_sources=sources,
        vanishing_x=vanishing_x,
        vanishing_y=vanishing_y,
        vanishing_z=vanishing_z,
        shared_edge_score_over_bbox_diag=score,
        bbox_diagonal_px=diag,
        gate_failures=tuple(dict.fromkeys(failures)),
    )


def _bbox(raw):
    if len(raw) != 4:
        raise Cube8TwoFaceGeometryError(
            "detector bbox must contain x1,y1,x2,y2")
    values = tuple(float(v) for v in raw)
    if not all(math.isfinite(v) for v in values):
        raise Cube8TwoFaceGeometryError(
            "detector bbox contains non-finite values")
    if values[2] <= values[0] or values[3] <= values[1]:
        raise Cube8TwoFaceGeometryError("detector bbox is invalid")
    return values


def _quad(raw, name):
    if len(raw) != 4:
        raise Cube8TwoFaceGeometryError(
            f"{name} must contain four cyclic vertices")
    points = tuple(_point(value, f"{name}[{i}]")
                   for i, value in enumerate(raw))
    if abs(_polygon_area2(points)) <= 1e-9:
        raise Cube8TwoFaceGeometryError(f"{name} is degenerate")
    if _proper_cross(
            points[0],
            points[1],
            points[2],
            points[3]) or _proper_cross(
            points[1],
            points[2],
            points[3],
            points[0]):
        raise Cube8TwoFaceGeometryError(
            f"{name} is self-crossing; vertices must be cyclic")
    return points


def _match_shared_edge(y, z, diag):
    best = None
    for yi in range(4):
        yj = (yi + 1) % 4
        for zi in range(4):
            zj = (zi + 1) % 4
            same = 0.5 * (_distance(y[yi], z[zi]) +
                          _distance(y[yj], z[zj])) / diag
            rev = 0.5 * (_distance(y[yi], z[zj]) +
                         _distance(y[yj], z[zi])) / diag
            candidate = (
                same, yi, yj, zi, zj, False) if same <= rev else (
                rev, yi, yj, zi, zj, True)
            if best is None or candidate[0] < best[0]:
                best = candidate
    if best is None:
        raise Cube8TwoFaceGeometryError("shared edge cannot be matched")
    return best


def _other_neighbor(index, shared_neighbor):
    first, second = (index - 1) % 4, (index + 1) % 4
    if first == shared_neighbor:
        return second
    if second == shared_neighbor:
        return first
    raise Cube8TwoFaceGeometryError("matched shared vertices are not an edge")


def _point(raw, name):
    if len(raw) < 2:
        raise Cube8TwoFaceGeometryError(f"{name} must contain x,y")
    result = (float(raw[0]), float(raw[1]))
    if not all(math.isfinite(v) for v in result):
        raise Cube8TwoFaceGeometryError(f"{name} is non-finite")
    return result


def _required(points, index):
    point = points[index]
    if point is None:
        raise Cube8TwoFaceGeometryError(f"required point {index} is missing")
    return point


def _average(a, b):
    return (0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]))


def _distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _bbox_interiority(point, bbox):
    x, y = point
    x1, y1, x2, y2 = bbox
    return min(x - x1, x2 - x, y - y1, y2 - y)


def _polygon_area2(points):
    return sum(
        points[i][0] * points[(i + 1) % 4][1] - points[(i + 1) % 4][0] * points[i][1]
        for i in range(4)
    )


def _line(a, b):
    x0, y0 = a
    x1, y1 = b
    A, B, C = y0 - y1, x1 - x0, x0 * y1 - x1 * y0
    norm = math.hypot(A, B)
    if norm <= 1e-12:
        raise Cube8TwoFaceGeometryError("projective line is degenerate")
    return (A / norm, B / norm, C / norm)


def _least_squares_intersection(lines):
    if len(lines) < 2:
        raise Cube8TwoFaceGeometryError(
            "line intersection requires at least two lines")
    aa = sum(a * a for a, _, _ in lines)
    ab = sum(a * b for a, b, _ in lines)
    bb = sum(b * b for _, b, _ in lines)
    ac = sum(a * c for a, _, c in lines)
    bc = sum(b * c for _, b, c in lines)
    det = aa * bb - ab * ab
    if abs(det) <= 1e-12:
        raise Cube8TwoFaceGeometryError(
            "projective line family is rank-deficient")
    point = ((-ac * bb + ab * bc) / det, (-aa * bc + ab * ac) / det)
    if not all(math.isfinite(v) for v in point):
        raise Cube8TwoFaceGeometryError(
            "projective intersection is non-finite")
    return point


def _orientation(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _proper_cross(a, b, c, d):
    o1, o2 = _orientation(a, b, c), _orientation(a, b, d)
    o3, o4 = _orientation(c, d, a), _orientation(c, d, b)
    eps = 1e-9
    return (o1 > eps and o2 < -eps or o1 < -eps and o2 > eps) and (
        o3 > eps and o4 < -eps or o3 < -eps and o4 > eps
    )
