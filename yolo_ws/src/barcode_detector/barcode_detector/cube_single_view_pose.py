"""Single-view cube pose from the visible face, with the face colour fixing yaw.

At the real camera mount (low, level, about the cube's height) only one face is
visible, and its four corners are coplanar. Coplanar PnP has a mirror ambiguity
and says nothing about *which* face it is, so a plain quad is not a pose. Two
extra pieces of evidence resolve it, both already in this package:

* the face's colour label (``cube_face_colors.classify_lab``) says which of the
  six faces is visible, and
* the cube's own frame is fixed once that face is known: the four in-plane
  rotations of the face are the only remaining freedom.

So: enumerate 6 faces x 4 in-plane rotations = 24 hypotheses, solve each with
IPPE (the square-target solver, which is what a cube face is), keep the ones that
reproject, and let the colour label choose. When the label abstains the pose is
returned with ``color_fixed_semantics_resolved`` false and the candidates listed,
which is the honest answer: position is good, yaw is not decided.
"""
from __future__ import annotations

import math

import numpy as np

from .cube_face_colors import FACE_NORMALS, classify_lab
from .cube_face_colors import _face_corner_object_points

FACE_NAMES = tuple(FACE_NORMALS.keys())


class FacePoseError(ValueError):
    """Raised when no face hypothesis explains the observation."""


def face_quad_corners(image_bgr, bbox_xyxy):
    """Corners of the visible face, ordered counter-clockwise from the top left."""
    import cv2

    image = np.asarray(image_bgr)
    x0, y0, x1, y1 = (int(round(float(value))) for value in bbox_xyxy)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(image.shape[1], x1), min(image.shape[0], y1)
    if x1 - x0 < 6 or y1 - y0 < 6:
        raise FacePoseError('bbox is too small to find a face')
    crop = image[y0:y1, x0:x1]
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    rows = max(1, int(lab.shape[0] * 0.15))
    background = np.median(lab[-rows:, :, :].reshape(-1, 3), axis=0)
    distance = np.linalg.norm(lab - background, axis=2)
    mask = (distance > 25.0).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise FacePoseError('no face silhouette inside the box')
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 40.0:
        raise FacePoseError('face silhouette is too small')
    perimeter = cv2.arcLength(contour, True)
    quad = cv2.approxPolyDP(contour, 0.03 * perimeter, True).reshape(-1, 2)
    if len(quad) != 4:
        box = cv2.boxPoints(cv2.minAreaRect(contour))
        quad = box
    quad = np.asarray(quad, dtype=np.float64) + np.asarray([x0, y0])
    quad = np.asarray(sorted(quad.tolist(), key=lambda point: math.atan2(
        point[1] - quad[:, 1].mean(), point[0] - quad[:, 0].mean())))
    return quad


def face_lab(image_bgr, bbox_xyxy, radius_fraction=0.2):
    """Median Lab of the face's middle, which is what the colour classifier reads."""
    image = np.asarray(image_bgr)
    x0, y0, x1, y1 = (int(round(float(value))) for value in bbox_xyxy)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(image.shape[1], x1), min(image.shape[0], y1)
    centre_u = (x0 + x1) // 2
    centre_v = (y0 + y1) // 2
    radius = max(2, int(radius_fraction * min(x1 - x0, y1 - y0)))
    patch = image[max(0, centre_v - radius):centre_v + radius + 1,
                  max(0, centre_u - radius):centre_u + radius + 1]
    if patch.size == 0:
        raise FacePoseError('no pixels to sample for the face colour')
    import cv2
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32)
    return tuple(float(value) for value in np.median(
        lab.reshape(-1, 3), axis=0))


def rotate_in_plane(corners, axis, quarter_turns):
    """Rotate face corners about their normal by 90-degree steps."""
    turns = int(quarter_turns) % 4
    others = [index for index in range(3) if index != axis]
    first, second = others
    result = []
    for point in corners:
        values = list(point)
        for _ in range(turns):
            a, b = values[first], values[second]
            values[first], values[second] = -b, a
        result.append(tuple(values))
    return result


def estimate_face_pose(
    image_bgr,
    camera_matrix,
    bbox_xyxy,
    *,
    side_length_m=0.057,
    distortion=(),
    maximum_reprojection_error_px=4.0,
):
    """Return (pose, evidence) for the face visible inside the box."""
    import cv2

    corners = face_quad_corners(image_bgr, bbox_xyxy)
    lab = face_lab(image_bgr, bbox_xyxy)
    label = classify_lab(lab)
    k = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    image_points = corners.astype(np.float64)

    hypotheses = []
    for face_name in FACE_NAMES:
        corners_object, axes = _face_corner_object_points(face_name,
                                                          side_length_m)
        axis = [index for index in range(3)
                if abs(FACE_NORMALS[face_name][index]) > 0.5][0]
        for turns in range(4):
            object_points = np.asarray(
                rotate_in_plane(corners_object, axis, turns), dtype=np.float64)
            try:
                count, rvecs, tvecs, _reprojection = cv2.solvePnPGeneric(
                    object_points, image_points, k, np.asarray(distortion or ()),
                    flags=cv2.SOLVEPNP_IPPE)
            except cv2.error:
                continue
            if count == 0:
                continue
            # IPPE returns the two planar solutions; keep the one in front of the
            # camera. The mirrored one has a negative z and would otherwise be
            # selected whenever it reprojects marginally better (measured: a pose
            # at z = -0.489 for a cube rendered at +0.5 m).
            best_solution = None
            for index in range(count):
                if float(tvecs[index].reshape(-1)[2]) <= 0.0:
                    continue
                projected, _ = cv2.projectPoints(object_points, rvecs[index],
                                                 tvecs[index], k,
                                                 np.asarray(distortion or ()))
                error = float(np.mean(np.linalg.norm(
                    projected.reshape(-1, 2) - image_points, axis=1)))
                if best_solution is None or error < best_solution[0]:
                    best_solution = (error, rvecs[index], tvecs[index])
            if best_solution is None:
                continue
            rvec, tvec = best_solution[1], best_solution[2]
            projected, _ = cv2.projectPoints(object_points, rvec, tvec, k,
                                             np.asarray(distortion or ()))
            error = float(np.mean(np.linalg.norm(
                projected.reshape(-1, 2) - image_points, axis=1)))
            if error > float(maximum_reprojection_error_px):
                continue
            rotation, _ = cv2.Rodrigues(rvec)
            orientation = _quaternion_from_matrix(rotation)
            hypotheses.append({
                'face': face_name,
                'quarter_turns': turns,
                'reprojection_error_px': error,
                'position_m': [float(value) for value in tvec.reshape(-1)],
                'orientation_xyzw': orientation,
                'color_label': label,
                'color_matches': label is not None and label == face_name,
            })

    if not hypotheses:
        raise FacePoseError('no face hypothesis reprojects within '
                            '%.1f px' % maximum_reprojection_error_px)

    matching = [item for item in hypotheses if item['color_matches']]
    pool = matching or hypotheses
    pool.sort(key=lambda item: item['reprojection_error_px'])
    best = pool[0]
    evidence = {
        'face_color_label': label,
        'face_color_lab': [round(value, 1) for value in lab],
        'hypotheses': len(hypotheses),
        'color_matched': bool(matching),
        'color_fixed_semantics_resolved': bool(matching),
        'candidate_faces': sorted({item['face'] for item in pool}),
        'yaw_candidates_deg': sorted(90.0 * item['quarter_turns']
                                     for item in pool[:4]),
    }
    pose = {
        'geometry_id': 'face-quad-v1',
        'position_m': best['position_m'],
        'orientation_xyzw': best['orientation_xyzw'],
        'reprojection_error_px': best['reprojection_error_px'],
        'visible_face': best['face'],
        'quarter_turns': best['quarter_turns'],
        'color_fixed_semantics_resolved': bool(matching),
    }
    return pose, evidence


def _quaternion_from_matrix(matrix):
    trace = float(matrix[0][0] + matrix[1][1] + matrix[2][2])
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        return [float((matrix[2][1] - matrix[1][2]) / scale),
                float((matrix[0][2] - matrix[2][0]) / scale),
                float((matrix[1][0] - matrix[0][1]) / scale),
                0.25 * scale]
    if matrix[0][0] > matrix[1][1] and matrix[0][0] > matrix[2][2]:
        scale = math.sqrt(1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2]) * 2.0
        return [0.25 * scale,
                float((matrix[0][1] + matrix[1][0]) / scale),
                float((matrix[0][2] + matrix[2][0]) / scale),
                float((matrix[2][1] - matrix[1][2]) / scale)]
    if matrix[1][1] > matrix[2][2]:
        scale = math.sqrt(1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2]) * 2.0
        return [float((matrix[0][1] + matrix[1][0]) / scale), 0.25 * scale,
                float((matrix[1][2] + matrix[2][1]) / scale),
                float((matrix[0][2] - matrix[2][0]) / scale)]
    scale = math.sqrt(1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1]) * 2.0
    return [float((matrix[0][2] + matrix[2][0]) / scale),
            float((matrix[1][2] + matrix[2][1]) / scale), 0.25 * scale,
            float((matrix[1][0] - matrix[0][1]) / scale)]
