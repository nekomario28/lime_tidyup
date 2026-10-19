"""Geometric visibility for the eight canonical cube vertices.

A detector confidence says "this pixel is probably this vertex". Geometry says
whether the vertex *can be seen at all* from the estimated pose. The two are
separate evidence and neither substitutes for the other: a low-confidence
observation of a visible corner is still a visible corner, and a
high-confidence observation of a geometrically impossible corner is a
contradiction rather than a measurement.

The mask itself is a property of the pose and the camera, so it is computed
from the cube geometry rather than read off the detector. The approach is
adapted from the adaptive-vision cube domain model, which derives its visible
landmarks the same way (``CubePoseTracker::visible_correspondences``) and
forbids inferring unseen faces from appearance alone.

Assumption: the cube is convex and unoccluded by other objects, so every vertex
belonging to a camera-facing face is visible. External occlusion is out of
scope here and must be reported by the caller.
"""

from __future__ import annotations

import math

from .cube_pose_3d import CubePose3DError, estimate_cube_pose_3d

CUBE_SIDE_M = 0.057

#: Visibility could not be decided from the available evidence.
VISIBILITY_UNKNOWN = 'unknown'
#: Geometry says the vertex faces away from the camera this frame.
VISIBILITY_OCCLUDED = 'occluded'
#: Visible and measured above the confidence floor.
VISIBILITY_OBSERVED = 'observed'
#: Visible but not measured; must not be used as a PnP observation.
VISIBILITY_UNMEASURED = 'unmeasured'
#: Detected above the confidence floor while geometry says it is occluded.
VISIBILITY_CONTRADICTED = 'contradicted'

CANONICAL_VERTEX_SIGNS = (
    (-1, -1, -1),
    (+1, -1, -1),
    (+1, +1, -1),
    (-1, +1, -1),
    (-1, -1, +1),
    (+1, -1, +1),
    (+1, +1, +1),
    (-1, +1, +1),
)


class CubeVisibilityError(ValueError):
    """Raised when visibility evidence cannot be computed."""


def _normalize(values):
    norm = math.sqrt(sum(float(value) ** 2 for value in values))
    if not math.isfinite(norm) or norm < 1e-12:
        raise CubeVisibilityError('cannot normalize a zero vector')
    return tuple(float(value) / norm for value in values)


def _rotate_inverse(q, vector):
    """Rotate ``vector`` by the inverse of quaternion ``q`` (xyzw)."""
    x, y, z, w = (float(value) for value in q)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm < 1e-12:
        raise CubeVisibilityError('invalid orientation quaternion')
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    # Inverse rotation: conjugate quaternion, q^-1 = (-x, -y, -z, w).
    ux, uy, uz = -x, -y, -z
    vx, vy, vz = vector
    cross = (
        uy * vz - uz * vy,
        uz * vx - ux * vz,
        ux * vy - uy * vx,
    )
    dot = ux * vx + uy * vy + uz * vz
    uu = ux * ux + uy * uy + uz * uz
    return tuple(
        2.0 * dot * u + (w * w - uu) * v + 2.0 * w * c
        for u, v, c in zip((ux, uy, uz), (vx, vy, vz), cross)
    )


def camera_in_object_frame(position_m, orientation_xyzw):
    """Return the camera position expressed in the object frame."""
    if len(position_m) != 3:
        raise CubeVisibilityError('position must have three components')
    camera = (0.0, 0.0, 0.0)
    delta = tuple(camera[i] - float(position_m[i]) for i in range(3))
    return _rotate_inverse(orientation_xyzw, delta)


def visible_vertex_mask(
    position_m,
    orientation_xyzw,
    *,
    side_length_m=CUBE_SIDE_M
):
    """Return the per-vertex visibility mask for the eight canonical vertices.

    A vertex is visible when at least one of its three adjacent faces points
    towards the camera, that is when any axis sign ``s`` satisfies
    ``s * camera_axis > side_length / 2``.
    """
    if not math.isfinite(float(side_length_m)) or side_length_m <= 0.0:
        raise CubeVisibilityError('side_length_m must be positive')
    half = float(side_length_m) / 2.0
    camera = camera_in_object_frame(position_m, orientation_xyzw)
    if not all(math.isfinite(value) for value in camera):
        raise CubeVisibilityError('camera position is not finite')
    return tuple(
        any(sign * camera[axis] > half for axis, sign in enumerate(signs))
        for signs in CANONICAL_VERTEX_SIGNS
    )


def hidden_vertex_indices(position_m, orientation_xyzw, **kwargs):
    """Return the vertices no face of which faces the camera."""
    mask = visible_vertex_mask(position_m, orientation_xyzw, **kwargs)
    return tuple(index for index, visible in enumerate(mask) if not visible)


def classify_vertices(
    keypoint_confidences,
    visible_mask,
    *,
    minimum_confidence=0.5
):
    """Classify each vertex from geometry and measurement confidence.

    The returned states are never silently promoted: a visible vertex that was
    not measured stays ``unmeasured`` rather than becoming an observation, and
    a confident detection of an occluded vertex becomes ``contradicted`` rather
    than being averaged into the pose.
    """
    confidences = tuple(float(value) for value in keypoint_confidences)
    mask = tuple(bool(value) for value in visible_mask)
    if len(confidences) != len(mask):
        raise CubeVisibilityError(
            'confidence and visibility lengths must match'
        )
    floor = float(minimum_confidence)
    states = []
    for confidence, visible in zip(confidences, mask):
        measured = math.isfinite(confidence) and confidence >= floor
        if visible and measured:
            states.append(VISIBILITY_OBSERVED)
        elif visible:
            states.append(VISIBILITY_UNMEASURED)
        elif measured:
            states.append(VISIBILITY_CONTRADICTED)
        else:
            states.append(VISIBILITY_OCCLUDED)
    return tuple(states)


def observation_mask(keypoint_confidences, visible_mask, **kwargs):
    """Return which vertices may be used as PnP observations.

    Only ``observed`` vertices qualify. ``unmeasured`` and ``occluded`` are
    excluded by construction, and ``contradicted`` is excluded because the
    geometry and the detector disagree about whether the corner exists here.
    """
    states = classify_vertices(
        keypoint_confidences, visible_mask, **kwargs
    )
    return tuple(state == VISIBILITY_OBSERVED for state in states)


class VisibilityConsistentPose:
    """A pose accepted only if its own visibility prediction agrees with it."""

    __slots__ = (
        'pose',
        'visible_mask',
        'states',
        'excluded_indices',
        'violation_count',
        'reprojection_error_px',
        'inlier_count',
    )

    def __init__(
        self,
        pose,
        visible_mask,
        states,
        excluded_indices,
        violation_count,
        reprojection_error_px,
        inlier_count
    ):
        """Store an accepted pose and the evidence used to accept it."""
        self.pose = pose
        self.visible_mask = visible_mask
        self.states = states
        self.excluded_indices = excluded_indices
        self.violation_count = violation_count
        self.reprojection_error_px = reprojection_error_px
        self.inlier_count = inlier_count


def _confidences_with_excluded(confidences, excluded_index):
    values = list(confidences)
    if excluded_index is not None:
        values[excluded_index] = 0.0
    return tuple(values)


def estimate_visibility_consistent_pose(
    keypoints_xy,
    keypoint_confidences,
    camera_matrix,
    distortion=(),
    *,
    side_length_m=CUBE_SIDE_M,
    minimum_confidence=0.5,
    maximum_reprojection_error_px=4.0
):
    """Estimate a pose whose predicted visibility agrees with its observations.

    The plain confidence gate cannot tell a bad corner from an unmeasurable
    one. This tries the confidence-gated observation set and, when that set
    contradicts its own pose (an observed corner that geometry says is behind
    the cube), the variants with one contradicting corner removed. A candidate
    is accepted only with zero visibility violations; otherwise the evidence is
    insufficient and ``None`` is returned rather than a pose guessed from
    contradictory input.

    Candidates are tried in a fixed order, so the result is deterministic.
    """
    confidences = tuple(float(value) for value in keypoint_confidences)
    points = tuple(tuple(float(v) for v in point) for point in keypoints_xy)
    if len(points) != 8 or len(confidences) != 8:
        raise CubeVisibilityError('exactly 8 ordered keypoints are required')

    base = tuple(
        index
        for index, confidence in enumerate(confidences)
        if math.isfinite(confidence) and confidence >= minimum_confidence
    )
    if len(base) < 4:
        return None

    attempts = [None]
    attempts.extend(base)

    best = None
    for excluded_index in attempts:
        candidate_confidences = _confidences_with_excluded(
            confidences, excluded_index
        )
        try:
            pose = estimate_cube_pose_3d(
                points,
                candidate_confidences,
                camera_matrix,
                distortion,
                side_length_m=side_length_m,
                minimum_keypoint_confidence=minimum_confidence,
                maximum_reprojection_error_px=(
                    maximum_reprojection_error_px
                ),
            )
        except CubePose3DError:
            continue

        mask = visible_vertex_mask(
            pose.translation_m,
            pose.orientation_xyzw,
            side_length_m=side_length_m,
        )
        states = classify_vertices(
            candidate_confidences, mask,
            minimum_confidence=minimum_confidence,
        )
        violations = tuple(
            index for index, state in enumerate(states)
            if state in (VISIBILITY_CONTRADICTED, VISIBILITY_UNMEASURED)
        )
        if violations:
            continue

        excluded = () if excluded_index is None else (excluded_index,)
        candidate = VisibilityConsistentPose(
            pose=pose,
            visible_mask=mask,
            states=states,
            excluded_indices=excluded,
            violation_count=0,
            reprojection_error_px=pose.reprojection_error_px,
            inlier_count=pose.inlier_count,
        )
        if best is None:
            best = candidate
        elif candidate.reprojection_error_px < best.reprojection_error_px:
            best = candidate
        if best is not None and best.reprojection_error_px <= 1.0:
            break
    return best
