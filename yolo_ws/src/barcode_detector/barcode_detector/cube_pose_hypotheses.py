"""Enumerate the cube poses a colourless view cannot distinguish.

A cube is invariant under its own rotation group, so corner observations fix
the pose only up to the symmetries that keep the observed and hidden vertices
apart. For the three-face view that `ccc` approaches through, exactly three
poses reproduce the same pixels (identity plus two 120-degree rotations about
the hidden corner axis); for a two-face view, two poses 180 degrees apart; for
a frontal single-face view, four poses 90 degrees apart.

The residuals are exact, so no amount of solver quality separates them. Colour
does: the three visible faces take three different colours, and each hypothesis
predicts a different colour for each visible *local* face slot. That prediction
is what this module exposes, so a caller with colour evidence can select a
hypothesis and report the margin instead of silently keeping whichever pose the
solve happened to return.

Scope: geometry and the colour *contract* only. No image reading, no colour
classification, and no motion authority. A frontal single-face view has four
hypotheses with identical colour predictions, so it is reported as unresolved
rather than guessed.
"""

from __future__ import annotations

import itertools
import math

from .cube_pose_3d import CubePose3DError, estimate_cube_pose_3d
from .cube_visibility import (
    CUBE_SIDE_M,
    visible_vertex_mask,
)

FACE_NAMES = ('+x', '-x', '+y', '-y', '+z', '-z')

FACE_NORMALS = {
    '+x': (1.0, 0.0, 0.0),
    '-x': (-1.0, 0.0, 0.0),
    '+y': (0.0, 1.0, 0.0),
    '-y': (0.0, -1.0, 0.0),
    '+z': (0.0, 0.0, 1.0),
    '-z': (0.0, 0.0, -1.0),
}

#: Standard six-colour assignment, against which hypotheses are scored.
#: This is the same face-to-colour convention the generated pose cube uses
#: (:mod:`barcode_detector.rubik_pose_cube`), and a test pins the two together:
#: if they disagree, every identity hypothesis scores as a mismatch.
#: A caller may pass any label set, provided opposite faces differ.
STANDARD_FACE_COLORS = {
    '+x': 'red',
    '-x': 'orange',
    '+y': 'blue',
    '-y': 'green',
    '+z': 'white',
    '-z': 'yellow',
}


def _rotation_group():
    """The 24 proper rotations of the cube as nested tuples (deterministic)."""
    group = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            rows = []
            for row, column in enumerate(perm):
                row_values = [0.0, 0.0, 0.0]
                row_values[column] = float(signs[row])
                rows.append(tuple(row_values))
            if _determinant(rows) > 0.5:
                group.append(tuple(rows))
    group.sort()
    return tuple(group)


def _determinant(m):
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


CUBE_SYMMETRIES = _rotation_group()


def _matmul(a, b):
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )


def _matvec(m, v):
    return tuple(sum(m[i][k] * v[k] for k in range(3)) for i in range(3))


def _transpose(m):
    return tuple(tuple(m[j][i] for j in range(3)) for i in range(3))


def _quat_to_matrix(q):
    x, y, z, w = (float(value) for value in q)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm < 1e-12:
        raise CubePose3DError('invalid pose quaternion')
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )


def _matrix_to_quat(m):
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        return ((m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s,
                (m[1][0] - m[0][1]) / s, 0.25 * s)
    if m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        return (0.25 * s, (m[0][1] + m[1][0]) / s,
                (m[0][2] + m[2][0]) / s, (m[2][1] - m[1][2]) / s)
    if m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        return ((m[0][1] + m[1][0]) / s, 0.25 * s,
                (m[1][2] + m[2][1]) / s, (m[0][2] - m[2][0]) / s)
    s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
    return ((m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s,
            0.25 * s, (m[1][0] - m[0][1]) / s)


def rotation_angle_deg(first, second):
    """Angle in degrees of the rotation taking ``first`` onto ``second``."""
    a = _quat_to_matrix(first)
    b = _quat_to_matrix(second)
    relative = _matmul(_transpose(a), b)
    cosine = (_trace(relative) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _trace(m):
    return m[0][0] + m[1][1] + m[2][2]


def visible_face_names(position_m, orientation_xyzw, *, side_length_m):
    """Faces of the *local* frame that point at the camera."""
    rotation = _quat_to_matrix(orientation_xyzw)
    camera_object = _matvec(
        _transpose(rotation),
        tuple(-float(value) for value in position_m),
    )
    half = float(side_length_m) / 2.0
    names = []
    for name in FACE_NAMES:
        normal = FACE_NORMALS[name]
        if sum(normal[i] * camera_object[i] for i in range(3)) > half:
            names.append(name)
    return tuple(names)


class PoseHypothesis:
    """One symmetry-equivalent pose and its predicted visible colours."""

    __slots__ = ('pose', 'symmetry', 'predicted_colors', 'visible_slots')

    def __init__(self, pose, symmetry, predicted_colors, visible_slots):
        """Store a hypothesis with the evidence needed to score it."""
        self.pose = pose
        self.symmetry = symmetry
        self.predicted_colors = predicted_colors
        self.visible_slots = visible_slots

    def color_assignment(self):
        """Return the slot-to-colour mapping this hypothesis predicts."""
        return tuple(zip(self.visible_slots, self.predicted_colors))


def _hidden_vertices(position_m, orientation_xyzw, side_length_m):
    mask = visible_vertex_mask(
        position_m, orientation_xyzw, side_length_m=side_length_m
    )
    return tuple(index for index, visible in enumerate(mask) if not visible)


def _vertex_signs():
    return tuple(
        (sx, sy, sz)
        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
    )


def enumerate_pose_hypotheses(
    keypoints_xy,
    keypoint_confidences,
    camera_matrix,
    distortion=(),
    *,
    side_length_m=CUBE_SIDE_M,
    minimum_confidence=0.5,
    maximum_reprojection_error_px=4.0,
    face_colors=None,
):
    """Return every pose that explains the observations equally well.

    The base pose comes from :func:`estimate_cube_pose_3d`; alternatives are
    that pose composed with each cube symmetry whose relabelling keeps the
    hidden vertices hidden. Returns a tuple ordered deterministically.
    """
    colors = dict(STANDARD_FACE_COLORS if face_colors is None else face_colors)
    missing = [name for name in FACE_NAMES if name not in colors]
    if missing:
        raise CubePose3DError(f'missing face colours: {missing}')

    base = estimate_cube_pose_3d(
        keypoints_xy,
        keypoint_confidences,
        camera_matrix,
        distortion,
        side_length_m=side_length_m,
        minimum_keypoint_confidence=minimum_confidence,
        maximum_reprojection_error_px=maximum_reprojection_error_px,
    )
    return hypotheses_from_pose(
        base.translation_m,
        base.orientation_xyzw,
        side_length_m=side_length_m,
        face_colors=colors,
    )


def hypotheses_from_pose(
    position_m,
    orientation_xyzw,
    *,
    side_length_m=CUBE_SIDE_M,
    face_colors=None,
):
    """Return the symmetry hypotheses of an already-solved pose.

    Used by the runtime so the visibility-consistent pose and the colour
    hypotheses share one solve and cannot disagree about the base pose.
    """
    colors = dict(STANDARD_FACE_COLORS if face_colors is None else face_colors)
    missing = [name for name in FACE_NAMES if name not in colors]
    if missing:
        raise CubePose3DError(f'missing face colours: {missing}')
    base_rotation = _quat_to_matrix(orientation_xyzw)
    hidden = set(_hidden_vertices(
        position_m, orientation_xyzw,
        side_length_m=side_length_m,
    ))
    signs = _vertex_signs()
    hidden_signs = {signs[index] for index in hidden}
    slots = visible_face_names(
        position_m, orientation_xyzw,
        side_length_m=side_length_m,
    )

    hypotheses = []
    for symmetry in CUBE_SYMMETRIES:
        relabelled = {tuple(_matvec(symmetry, signs[i])) for i in hidden}
        if not relabelled <= hidden_signs:
            continue
        rotation = _matmul(base_rotation, symmetry)
        quaternion = _matrix_to_quat(rotation)
        predicted = tuple(
            colors[_face_name_of(_matvec(symmetry, FACE_NORMALS[slot]))]
            for slot in slots
        )
        hypotheses.append(
            PoseHypothesis(
                pose=(tuple(float(v) for v in position_m), quaternion),
                symmetry=symmetry,
                predicted_colors=predicted,
                visible_slots=slots,
            )
        )
    hypotheses.sort(key=lambda item: item.pose[1])
    return tuple(hypotheses)


def _face_name_of(vector):
    for name in FACE_NAMES:
        normal = FACE_NORMALS[name]
        if all(abs(normal[i] - vector[i]) < 1e-9 for i in range(3)):
            return name
    raise CubePose3DError('vector is not an axis-aligned face normal')


class ColorSelection:
    """Outcome of scoring hypotheses against observed face colours."""

    __slots__ = ('hypothesis', 'observed', 'agreement', 'runner_up', 'margin')

    def __init__(self, hypothesis, observed, agreement, runner_up, margin):
        """Store the selected hypothesis and why it won."""
        self.hypothesis = hypothesis
        self.observed = observed
        self.agreement = agreement
        self.runner_up = runner_up
        self.margin = margin


def select_by_face_colors(hypotheses, observed_colors):
    """Pick the only hypothesis whose colour prediction matches observation.

    ``observed_colors`` maps a visible *local* slot (``'+x'`` …) to a colour
    label. Every slot must be present; an unobserved face is missing evidence,
    not a wildcard, so a partial observation raises instead of being ignored.

    Returns ``None`` when the best agreement is not unique — including the
    frontal single-face case, where all four hypotheses predict the same colour
    — because that is exactly the ambiguity this module exists to report.
    """
    observed = dict(observed_colors)
    if not hypotheses:
        return None
    slots = hypotheses[0].visible_slots
    for hypothesis in hypotheses:
        missing = [slot for slot in slots if slot not in observed]
        if missing:
            raise CubePose3DError(f'missing observed face colours: {missing}')
    if any(observed[slot] is None for slot in slots):
        # An unlabelled face is missing evidence, not a mismatch: scoring it
        # as a wrong colour would let a partial observation win a hypothesis.
        return None

    scores = []
    for hypothesis in hypotheses:
        agreement = sum(
            1
            for slot, predicted in zip(slots, hypothesis.predicted_colors)
            if observed[slot] == predicted
        )
        scores.append((agreement, hypothesis))
    scores.sort(key=lambda item: -item[0])
    best_agreement, best = scores[0]
    if len(scores) > 1 and scores[1][0] == best_agreement:
        return None
    if best_agreement == 0:
        return None
    runner_up = scores[1][1] if len(scores) > 1 else None
    return ColorSelection(
        hypothesis=best,
        observed=observed,
        agreement=best_agreement,
        runner_up=runner_up,
        margin=best_agreement - (scores[1][0] if len(scores) > 1 else 0),
    )
