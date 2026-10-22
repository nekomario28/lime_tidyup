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
from .cube_pose_hypotheses import STANDARD_FACE_COLORS
from .cube_face_colors import _face_corner_object_points

FACE_NAMES = tuple(FACE_NORMALS.keys())

#: The coloured plates cover 3 facelets on a pitch of 0.0174 m with 0.0162 m
#: plates, i.e. 0.051 m of a 0.057 m face (rubik_pose_cube: FACELET_SIZE_M,
#: FACELET_GAP_M). The detected quad is the *plate field*, not the face's outer
#: edge, so solving it as a full face overestimates the distance by 1/0.895 =
#: 11.7 percent -- measured as 2.5 cm at 0.4 m growing to 10.5 cm at 1.0 m.
STICKER_FRACTION = 0.895


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


#: OpenCV hue centres for the cube colours (red wraps through 0).
HUE_CENTRES = {'red': 3.0, 'orange': 16.0, 'yellow': 30.0, 'green': 60.0,
               'blue': 110.0}
#: Kept at the blob source's tolerance. Tightening it to 8 to stop red (3) and
#: orange (16) overlapping was measured and is worse: the mask loses too many
#: plates, the quad becomes unstable and the face is misidentified (27 cm at 1 m
#: against 6.6 cm at 12).
HUE_TOLERANCE = 12.0
MINIMUM_SATURATION = 90
MINIMUM_VALUE = 40


def face_quad_from_color(image_bgr, bbox_xyxy, color):
    """Tight quad of one colour's region inside the box, without the blob margin.

    The frontend fits a face to its own colour mask, but it needs two visible faces
    to bootstrap and refuses when only one is visible -- which is the real mount's
    situation. This is the same idea for a single named colour: mask that hue,
    take the largest component, and fit its corners.
    """
    import cv2

    centre = HUE_CENTRES.get(color)
    if centre is None:
        return None
    image = np.asarray(image_bgr)
    x0, y0, x1, y1 = (int(round(float(value))) for value in bbox_xyxy)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(image.shape[1], x1), min(image.shape[0], y1)
    if x1 - x0 < 6 or y1 - y0 < 6:
        return None
    crop = image[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.float32)
    distance = np.minimum(np.abs(hue - centre), 180.0 - np.abs(hue - centre))
    mask = ((hsv[:, :, 1] >= MINIMUM_SATURATION)
            & (hsv[:, :, 2] >= MINIMUM_VALUE)
            & (distance <= HUE_TOLERANCE)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 40.0:
        return None
    perimeter = cv2.arcLength(contour, True)
    quad = cv2.approxPolyDP(contour, 0.03 * perimeter, True).reshape(-1, 2)
    if len(quad) != 4:
        quad = cv2.boxPoints(cv2.minAreaRect(contour))
    return np.asarray(quad, dtype=np.float64) + np.asarray([x0, y0])


def measure_sticker_span(image_bgr, bbox_xyxy, corners, side_length_m):
    """Physical size of the detected colour region, measured from the frame.

    The detector reports the sticker field, and the model says that field is 0.051
    of a 0.057 face -- but solving at that fixed ratio left a residual that behaved
    like a scale error (measured 4-7 cm), because the mask can merge plates from two
    faces. The ratio is measurable instead of assumed: a fronto-parallel cube's
    silhouette is exactly one side wide, so

        span = side * (colour quad width in px) / (object silhouette width in px)

    No search is needed. Returns a span in metres, clamped to a plausible band so a
    bad silhouette cannot produce a nonsense pose.
    """
    import cv2

    try:
        from .cube_bbox_source import colour_blob_bbox  # noqa: F401  (rule reference)
        from .cube_pose_tracking import colour_mask, object_mask
    except Exception:  # noqa: BLE001
        return None
    try:
        seed, _area = colour_mask(image_bgr)
        mask, _object_area = object_mask(image_bgr, seed)
    except Exception:  # noqa: BLE001
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    object_box = cv2.boundingRect(contour)
    object_width = float(object_box[2])
    if object_width < 4.0:
        return None
    sides = [float(np.linalg.norm(corners[(index + 1) % 4] - corners[index]))
             for index in range(4)]
    quad_width = 0.5 * (sides[0] + sides[2])
    ratio = quad_width / object_width
    span = float(side_length_m) * ratio
    lower, upper = 0.5 * float(side_length_m), float(side_length_m)
    return min(upper, max(lower, span))


def estimate_face_pose(
    image_bgr,
    camera_matrix,
    bbox_xyxy,
    *,
    side_length_m=0.057,
    sticker_fraction=STICKER_FRACTION,
    use_measured_span=False,
    distortion=(),
    maximum_reprojection_error_px=4.0,
):
    """Return (pose, evidence) for the face visible inside the box."""
    import cv2

    lab = face_lab(image_bgr, bbox_xyxy)
    label = classify_lab(lab)
    measured_span = None
    # quad_from_frontend is not called: the two-face frontend refuses when only one
    # face is visible, which is this mount's normal case, and running it per frame
    # only costs time. The colour's own mask is the tight quad here.
    corners = face_quad_from_color(image_bgr, bbox_xyxy, label) if label else None
    quad_source = 'face_color_mask' if corners is not None else 'blob_mask'
    if corners is None:
        corners = face_quad_corners(image_bgr, bbox_xyxy)

    # Measuring the span from the object silhouette was tried and is worse: the
    # silhouette is not a reliable width reference at this mount (measured 12-37 cm
    # error, growing with distance, because the object mask is wider than the cube
    # face -- it takes in the shadow and the arm). Kept behind a flag for anyone
    # with a cleaner segmentation; the fixed plate-field ratio is the default.
    measured_span = None
    if use_measured_span:
        measured_span = measure_sticker_span(image_bgr, bbox_xyxy, corners,
                                             side_length_m)
        if measured_span is not None:
            sticker_fraction = measured_span / float(side_length_m)
    k = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    image_points = corners.astype(np.float64)

    hypotheses = []
    for face_name in FACE_NAMES:
        corners_object, axes = _face_corner_object_points(face_name,
                                                          side_length_m)
        axis = [index for index in range(3)
                if abs(FACE_NORMALS[face_name][index]) > 0.5][0]
        # shrink the in-plane axes to the plate field, keeping the cube centre at
        # the origin and the face's own plane where it is
        corners_object = [
            tuple(value * float(sticker_fraction) if index != axis else value
                  for index, value in enumerate(point))
            for point in corners_object]
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
            # A face whose normal points away from the camera cannot be the one we
            # see: drop it before the colour is consulted, so the colour only has
            # to choose between the faces that face us.
            normal_object = np.asarray(FACE_NORMALS[face_name], dtype=np.float64)
            normal_camera = rotation @ normal_object
            to_camera = -np.asarray(tvec.reshape(-1), dtype=np.float64)
            if float(np.dot(normal_camera, to_camera)) <= 0.0:
                continue
            orientation = _quaternion_from_matrix(rotation)
            hypotheses.append({
                'face': face_name,
                'face_color': STANDARD_FACE_COLORS.get(face_name),
                'quarter_turns': turns,
                'reprojection_error_px': error,
                'position_m': [float(value) for value in tvec.reshape(-1)],
                'orientation_xyzw': orientation,
                'color_label': label,
                # label is a colour ("orange"), face is an axis ("-x"): compare
                # through the colour convention, not directly (that mistake made
                # every hypothesis look unmatched).
                'color_matches': (label is not None
                                  and STANDARD_FACE_COLORS.get(face_name) == label),
            })

    if not hypotheses:
        raise FacePoseError('no face hypothesis reprojects within '
                            '%.1f px' % maximum_reprojection_error_px)

    # Colour first, geometry second: the label says which face is visible, and only
    # then does reprojection choose among that face's in-plane rotations. Selecting
    # on reprojection first let a wrong face survive (measured: -y chosen while the
    # colour said orange).
    matching = [item for item in hypotheses if item['color_matches']]
    pool = matching or hypotheses
    pool.sort(key=lambda item: item['reprojection_error_px'])
    best = pool[0]
    evidence = {
        'quad_source': quad_source,
        'sticker_span_m': (None if measured_span is None
                           else round(float(measured_span), 4)),
        'sticker_fraction': round(float(sticker_fraction), 3),
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
