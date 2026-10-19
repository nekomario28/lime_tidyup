"""Assemble the pose evidence a cube observation is allowed to carry.

Pure functions only: no ROS, no parameters, no logging. The node calls this
with an image, a camera model and one detection, and gets the detection back
with the evidence it actually earned:

* a geometrically consistent pose (visibility gate), and
* whether the pose's *colour* semantics are resolved, the flag that keeps
  Cube8 grasp candidates withheld today.

Both stages are opt-in, so the existing `/cube_pose_result` contract is
unchanged when they are off. Nothing here grants motion authority: a resolved
pose is still an observation; accept/withhold stays downstream.
"""

from __future__ import annotations

from .cube_face_colors import observe_face_colors
from .cube_pose_3d import GEOMETRY_ID
from .cube_pose_hypotheses import (
    hypotheses_from_pose,
    select_by_face_colors,
)
from .cube_visibility import estimate_visibility_consistent_pose


UNRESOLVED_COLOR_REASON = 'visible face colours are not uniquely resolved'


def _camera_matrix(camera_info):
    return tuple(float(value) for value in camera_info.k)


def _distortion(camera_info):
    return tuple(float(value) for value in getattr(camera_info, 'd', ()))


def _pose_dict(pose, frame_id, timestamp, side_length_m):
    return pose.as_dict(
        frame_id=frame_id,
        timestamp=timestamp,
        side_length_m=side_length_m,
    )


def _hypothesis_pose_dict(
    hypothesis,
    base,
    frame_id,
    timestamp,
    side_length_m,
    selection,
    observed,
):
    """Return a pose dict for a symmetry hypothesis chosen by colour.

    The reprojection evidence is inherited from the base pose because every
    symmetry hypothesis reproduces the same pixels exactly; only the colour
    evidence separates them, so the margin is what belongs in the record.
    """
    position, quaternion = hypothesis.pose
    return {
        'geometry_id': GEOMETRY_ID,
        'frame_id': frame_id,
        'timestamp': timestamp,
        'side_length_m': float(side_length_m),
        'position_m': [float(value) for value in position],
        'orientation_xyzw': [float(value) for value in quaternion],
        'reprojection_error_px': base['reprojection_error_px'],
        'correspondence_count': base['correspondence_count'],
        'inlier_count': base['inlier_count'],
        'inlier_mask': list(base['inlier_mask']),
        'color_fixed_semantics_resolved': True,
        'hypothesis_color_margin': int(selection.margin),
        'hypothesis_color_agreement': int(selection.agreement),
        'visible_face_colors': dict(selection.observed),
    }


def attach_pose_evidence(
    image,
    camera_info,
    detection,
    *,
    side_length_m,
    frame_id,
    timestamp,
    enable_visibility_gate=False,
    enable_face_color_selection=False,
    minimum_keypoint_confidence=0.5,
    maximum_reprojection_error_px=4.0,
    minimum_chroma=None,
    minimum_hue_margin_deg=None,
):
    """Return ``detection`` with visibility and colour evidence attached.

    When both stages are disabled the input is returned untouched, so callers
    that keep the default parameters see exactly the old contract.
    """
    if not enable_visibility_gate and not enable_face_color_selection:
        return detection

    evidence = dict(detection.get('pose_evidence') or {})
    keypoints = tuple(
        tuple(float(value) for value in point)
        for point in detection.get('keypoints_xy') or ()
    )
    confidences = tuple(
        float(value)
        for value in detection.get('keypoints_confidence') or ()
    )
    pose = detection.get('pose_3d')

    if enable_visibility_gate:
        if len(keypoints) != 8 or len(confidences) != 8:
            evidence['visibility'] = {
                'status': 'skipped',
                'reason': 'exactly 8 ordered keypoints are required',
                'keypoint_count': len(keypoints),
            }
        else:
            consistent = estimate_visibility_consistent_pose(
                keypoints,
                confidences,
                _camera_matrix(camera_info),
                _distortion(camera_info),
                side_length_m=side_length_m,
                minimum_confidence=minimum_keypoint_confidence,
                maximum_reprojection_error_px=(
                    maximum_reprojection_error_px
                ),
            )
            if consistent is None:
                evidence['visibility'] = {
                    'status': 'rejected',
                    'reason': 'observations contradict the predicted '
                              'visibility',
                }
                detection = dict(detection)
                detection.pop('pose_3d', None)
                detection['pose_evidence'] = evidence
                return detection
            pose = _pose_dict(
                consistent.pose, frame_id, timestamp, side_length_m
            )
            pose['visibility_excluded_indices'] = list(
                consistent.excluded_indices
            )
            pose['visible_vertex_count'] = int(
                sum(1 for visible in consistent.visible_mask if visible)
            )
            evidence['visibility'] = {
                'status': 'accepted',
                'excluded_indices': list(consistent.excluded_indices),
                'violation_count': int(consistent.violation_count),
            }
            detection = dict(detection)
            detection['pose_3d'] = pose

    if enable_face_color_selection:
        evidence['color'] = _attach_color_evidence(
            image,
            camera_info,
            detection,
            pose=pose,
            side_length_m=side_length_m,
            frame_id=frame_id,
            timestamp=timestamp,
            minimum_chroma=minimum_chroma,
            minimum_hue_margin_deg=minimum_hue_margin_deg,
        )

    detection = dict(detection)
    detection['pose_evidence'] = evidence
    return detection


def _attach_color_evidence(
    image,
    camera_info,
    detection,
    *,
    pose,
    side_length_m,
    frame_id,
    timestamp,
    minimum_chroma,
    minimum_hue_margin_deg,
):
    if pose is None:
        return {'status': 'unavailable', 'reason': 'no accepted pose'}
    if image is None:
        return {'status': 'unavailable', 'reason': 'no image'}

    kwargs = {}
    if minimum_chroma is not None:
        kwargs['minimum_chroma'] = float(minimum_chroma)
    if minimum_hue_margin_deg is not None:
        kwargs['minimum_hue_margin_deg'] = float(minimum_hue_margin_deg)

    hypotheses = hypotheses_from_pose(
        pose['position_m'],
        pose['orientation_xyzw'],
        side_length_m=side_length_m,
    )
    slots = hypotheses[0].visible_slots
    observed, colors_evidence = observe_face_colors(
        image,
        pose['position_m'],
        pose['orientation_xyzw'],
        _camera_matrix(camera_info),
        slots,
        side_length_m=side_length_m,
        **kwargs
    )

    unresolved = [slot for slot, label in observed.items() if label is None]
    if unresolved:
        pose['color_fixed_semantics_resolved'] = False
        pose['visible_face_colors'] = dict(observed)
        return {
            'status': 'unresolved',
            'reason': UNRESOLVED_COLOR_REASON,
            'unresolved_slots': unresolved,
            'faces': colors_evidence,
            'hypothesis_count': len(hypotheses),
        }

    selection = select_by_face_colors(hypotheses, observed)
    if selection is None:
        pose['color_fixed_semantics_resolved'] = False
        pose['visible_face_colors'] = dict(observed)
        return {
            'status': 'unresolved',
            'reason': UNRESOLVED_COLOR_REASON,
            'faces': colors_evidence,
            'hypothesis_count': len(hypotheses),
        }

    detection['pose_3d'] = _hypothesis_pose_dict(
        selection.hypothesis,
        pose,
        frame_id,
        timestamp,
        side_length_m,
        selection,
        observed,
    )
    return {
        'status': 'resolved',
        'faces': colors_evidence,
        'hypothesis_count': len(hypotheses),
        'agreement': int(selection.agreement),
        'margin': int(selection.margin),
    }
