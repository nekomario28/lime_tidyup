"""Model-free cube pose: geometry and colour, no learned checkpoint.

This is a second perception route next to the YOLO/RTMPose keypoint path. It
needs a detector only for a bounding box; the pose itself comes from geometry:

1. two visible face regions are found by hue/white thresholds inside the box;
2. their quads are refined onto the object silhouette (the colour regions are
   inset by the sticker gaps and the core rim, measured at 4-17 px);
3. the view-local cube vertices are reconstructed from the two quads and solved
   with a label search over the corner-pair relabellings, so the pose is only
   accepted when exactly one relabelling fits the reprojection gate;
4. the face colours then select which symmetry-equivalent pose is meant, and
   that is what makes the pose colour-fixed rather than merely view-local.

Why keep it: the measured eight-keypoint checkpoint produced vertices that no
cube pose could explain (inlier RMS above a 30 px budget), while this route was
measured at a 1.31 px residual and a sub-millimetre translation on the same
frame. It also fails closed in a legible way: a single visible face cannot
separate the symmetry-equivalent poses and the colour leg abstains.

Provenance: the frontend, the view-local reconstruction and the PnP contract are
ported from this project's cube8 lane (`cube8_two_face_color_frontend`,
`cube8_two_face_view_local`, `cube8_view_local_pnp` and their dependencies), which
in turn adapts `nekomario28/lime-interactive-transport`.
"""

from __future__ import annotations

from dataclasses import dataclass

from .cube_bbox_source import BboxSourceError, choose_bbox
from .cube_face_colors import observe_face_colors
from .cube_pose_hypotheses import hypotheses_from_pose
from .cube8_pnp import CameraModel
from .cube8_two_face_color_frontend import (
    Cube8TwoFaceColorError,
    TwoFaceColorFrontendConfig,
    recover_two_face_view_local_from_bgr,
)
from .cube8_view_local_pnp import (
    Cube8PnpError as ViewLocalPnpError,
    estimate_view_local_pose_with_label_search,
)

UNRESOLVED_REASON = 'visible face colours are not uniquely resolved'
#: Which physical face each standard colour belongs to. The pose cube asset and
#: the hypothesis scorer use the same table, so a labelled face identifies an
#: axis and a sign.
FACE_BY_COLOR = {
    'red': '+x', 'orange': '-x',
    'blue': '+y', 'green': '-y',
    'white': '+z', 'yellow': '-z',
}
FACE_AXIS = {'x': 0, 'y': 1, 'z': 2}


class ModelFreePoseError(ValueError):
    """Raised when the geometry and colour evidence cannot produce a pose."""


@dataclass(frozen=True)
class ModelFreePose:
    """Accepted pose plus the evidence that made it acceptable."""

    translation_m: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    reprojection_error_px: float
    visible_face_colors: dict
    color_fixed_semantics_resolved: bool
    label_swaps: tuple[bool, bool, bool]
    visible_faces: tuple[str, ...]
    hypothesis_count: int
    color_agreement: int | None
    color_margin: int | None
    unresolved_reason: str
    face_evidence: dict
    correspondence_count: int
    inlier_count: int
    inlier_mask: tuple[int, ...]
    bbox_source: str = 'detector'
    bbox_quality: float = 1.0
    bbox_xyxy: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


def _sign_matrix(physical_by_slot):
    """Local slot -> physical face map, as a signed permutation matrix.

    Returns ``(matrix, determinant)``. The determinant says whether the labelled
    faces can be explained by a rigid pose: a proper rotation needs ``+1``, while
    ``-1`` means the observation describes a mirror of the cube and must not be
    turned into a pose.
    """
    matrix = [[0.0, 0.0, 0.0] for _ in range(3)]
    used = set()
    for slot, face in physical_by_slot.items():
        slot_axis = FACE_AXIS[slot[1]]
        face_axis = FACE_AXIS[face[1]]
        if face_axis in used:
            raise ModelFreePoseError(
                f'two visible slots map to the same physical axis: {face}'
            )
        used.add(face_axis)
        matrix[face_axis][slot_axis] = 1.0 if face[0] == '+' else -1.0
    if len(used) != 3:
        raise ModelFreePoseError('all three visible faces must be labelled')
    determinant = (matrix[0][0] *
                   (matrix[1][1] *
                    matrix[2][2] -
                    matrix[1][2] *
                    matrix[2][1]) -
                   matrix[0][1] *
                   (matrix[1][0] *
                    matrix[2][2] -
                    matrix[1][2] *
                    matrix[2][0]) +
                   matrix[0][2] *
                   (matrix[1][0] *
                    matrix[2][1] -
                    matrix[1][1] *
                    matrix[2][0]))
    return matrix, determinant


def _compose(orientation_xyzw, sign_matrix):
    """Compose a local pose orientation with the local->physical map."""
    x, y, z, w = (float(v) for v in orientation_xyzw)
    norm = (x * x + y * y + z * z + w * w) ** 0.5
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    rotation = [[
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
    ], [
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
    ], [
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ]]
    product = [
        [sum(rotation[i][k] * sign_matrix[k][j] for k in range(3))
         for j in range(3)]
        for i in range(3)
    ]
    trace = product[0][0] + product[1][1] + product[2][2]
    if trace > 0:
        scale = (trace + 1.0) ** 0.5 * 2
        quaternion = (
            (product[2][1] - product[1][2]) / scale,
            (product[0][2] - product[2][0]) / scale,
            (product[1][0] - product[0][1]) / scale,
            0.25 * scale,
        )
    elif product[0][0] > product[1][1] and product[0][0] > product[2][2]:
        scale = (1.0 + product[0][0] - product[1]
                 [1] - product[2][2]) ** 0.5 * 2
        quaternion = (
            0.25 * scale,
            (product[0][1] + product[1][0]) / scale,
            (product[0][2] + product[2][0]) / scale,
            (product[2][1] - product[1][2]) / scale,
        )
    elif product[1][1] > product[2][2]:
        scale = (1.0 + product[1][1] - product[0]
                 [0] - product[2][2]) ** 0.5 * 2
        quaternion = (
            (product[0][1] + product[1][0]) / scale,
            0.25 * scale,
            (product[1][2] + product[2][1]) / scale,
            (product[0][2] - product[2][0]) / scale,
        )
    else:
        scale = (1.0 + product[2][2] - product[0]
                 [0] - product[1][1]) ** 0.5 * 2
        quaternion = (
            (product[0][2] + product[2][0]) / scale,
            (product[1][2] + product[2][1]) / scale,
            0.25 * scale,
            (product[1][0] - product[0][1]) / scale,
        )
    return tuple(float(v) for v in quaternion)


def estimate_model_free_pose(
    image,
    camera_matrix,
    bbox_xyxy=None,
    *,
    depth_image=None,
    bbox_source='detector',
    depth_config=None,
    roi_xyxy=None,
    side_length_m=0.057,
    maximum_reprojection_error_px=4.0,
    minimum_chroma=None,
    minimum_hue_margin_deg=None,
    config=None,
):
    """Estimate a colour-fixed cube pose from two visible faces.

    ``camera_matrix`` is the 3x3 intrinsics in row-major order (the ROS
    ``CameraInfo.k`` layout). Raises :class:`ModelFreePoseError` when the gates
    reject the observation; a pose whose colours are ambiguous is returned with
    ``color_fixed_semantics_resolved`` false and the reason recorded, because
    the geometry is still usable evidence.
    """
    try:
        bbox = choose_bbox(
            bbox_source,
            detector_box=bbox_xyxy,
            depth_image=depth_image,
            depth_config=depth_config,
            roi_xyxy=roi_xyxy,
        )
    except BboxSourceError as error:
        raise ModelFreePoseError(f'bbox source rejected: {error}') from error

    camera = CameraModel(
        k=tuple(float(value) for value in camera_matrix),
        d=(0.0, 0.0, 0.0, 0.0, 0.0),
    )
    frontend_config = config or TwoFaceColorFrontendConfig()
    try:
        observation = recover_two_face_view_local_from_bgr(
            image, bbox.box_xyxy, config=frontend_config
        )
    except Cube8TwoFaceColorError as error:
        raise ModelFreePoseError(
            f'two-face frontend rejected: {error}') from error

    geometry = observation.geometry
    if not geometry.gate_passed:
        raise ModelFreePoseError(
            'two-face geometry gate failed: '
            + ', '.join(geometry.gate_failures)
        )

    try:
        view_local = estimate_view_local_pose_with_label_search(
            geometry.points,
            camera,
            derived_vertex_indices=(1,),
            side_length_m=side_length_m,
            maximum_reprojection_error_px=maximum_reprojection_error_px,
        )
    except ViewLocalPnpError as error:
        raise ModelFreePoseError(
            f'view-local PnP rejected: {error}') from error

    pose = view_local.pose
    hypotheses = hypotheses_from_pose(
        pose.translation_m,
        pose.orientation_xyzw,
        side_length_m=side_length_m,
    )
    slots = hypotheses[0].visible_slots
    kwargs = {}
    if minimum_chroma is not None:
        kwargs['minimum_chroma'] = float(minimum_chroma)
    if minimum_hue_margin_deg is not None:
        kwargs['minimum_hue_margin_deg'] = float(minimum_hue_margin_deg)
    observed, face_evidence = observe_face_colors(
        image,
        pose.translation_m,
        pose.orientation_xyzw,
        camera_matrix,
        slots,
        side_length_m=side_length_m,
        **kwargs
    )

    labels_known = all(label is not None for label in observed.values())
    sign_matrix = None
    if labels_known:
        physical_by_slot = {
            slot: FACE_BY_COLOR[colour]
            for slot, colour in observed.items()
            if colour in FACE_BY_COLOR
        }
        if len(physical_by_slot) == len(observed):
            sign_matrix, determinant = _sign_matrix(physical_by_slot)
            if determinant < 0.0:
                raise ModelFreePoseError(
                    'visible face colours describe a mirrored cube; refusing '
                    'to convert them into a pose'
                )
    # Colour semantics are resolved by the face-sign map itself: once every
    # visible slot is labelled with a physical face, the remaining symmetry is
    # gone. No hypothesis competition is left to score, so no margin is
    # claimed.
    if sign_matrix is None:
        return ModelFreePose(
            translation_m=tuple(float(v) for v in pose.translation_m),
            orientation_xyzw=tuple(float(v) for v in pose.orientation_xyzw),
            reprojection_error_px=float(pose.reprojection_error_px),
            visible_face_colors=dict(observed),
            color_fixed_semantics_resolved=False,
            label_swaps=tuple(view_local.label_swaps),
            visible_faces=tuple(slots),
            hypothesis_count=len(hypotheses),
            color_agreement=None,
            color_margin=None,
            unresolved_reason=UNRESOLVED_REASON,
            face_evidence=face_evidence,
            correspondence_count=int(pose.correspondence_count),
            inlier_count=int(pose.inlier_count),
            inlier_mask=tuple(int(v) for v in pose.inlier_mask),
            bbox_source=bbox.source,
            bbox_quality=bbox.quality,
            bbox_xyxy=tuple(float(v) for v in bbox.box_xyxy),
        )

    # The colour labels say which physical face each visible slot is, so the
    # pose is lifted out of the view-local frame instead of being composed with
    # whichever symmetry the scorer happened to prefer.
    position = pose.translation_m
    quaternion = _compose(pose.orientation_xyzw, sign_matrix)
    return ModelFreePose(
        translation_m=tuple(float(v) for v in position),
        orientation_xyzw=quaternion,
        reprojection_error_px=float(pose.reprojection_error_px),
        visible_face_colors=dict(observed),
        color_fixed_semantics_resolved=True,
        label_swaps=tuple(view_local.label_swaps),
        visible_faces=tuple(slots),
        hypothesis_count=len(hypotheses),
        color_agreement=int(sum(
            1 for slot, label in observed.items()
            if FACE_BY_COLOR.get(label) == physical_by_slot.get(slot)
        )),
        color_margin=None,
        unresolved_reason='',
        face_evidence=face_evidence,
        correspondence_count=int(pose.correspondence_count),
        inlier_count=int(pose.inlier_count),
        inlier_mask=tuple(int(v) for v in pose.inlier_mask),
        bbox_source=bbox.source,
        bbox_quality=bbox.quality,
        bbox_xyxy=tuple(float(v) for v in bbox.box_xyxy),
    )
