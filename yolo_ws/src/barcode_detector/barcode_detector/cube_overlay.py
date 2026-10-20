"""Draw what the model-free cube perception actually recognised.

One pure function, no ROS, so the overlay is testable and so the node stays a
transport shell. It draws only measured things and labels them as such:

* the bounding box that was used, and which source produced it;
* the two visible face regions with the colour each was classified as (or
  ``?`` when the classifier abstained);
* the silhouette polygon the corners were snapped to;
* the cube pose reprojected as a wireframe with its axes, so a wrong pose is
  visible as a wireframe that does not sit on the cube;
* a status line: accepted/rejected and the reason, the residual, whether the
  colour semantics resolved, and the label relabelling that was chosen.

A rejected frame is still drawn. "Why is there no pose" is the question this is
most needed for, and an empty window cannot answer it.
"""

from __future__ import annotations

import math

import numpy as np

#: Fixed drawing palette (BGR), so the same meaning always has the same colour.
COLOR_ACCEPTED = (80, 255, 80)
COLOR_REJECTED = (60, 60, 255)
COLOR_PENDING = (0, 200, 255)
COLOR_BOX = (255, 190, 0)
COLOR_SILHOUETTE = (200, 200, 200)
COLOR_FACE = (0, 255, 255)
COLOR_AXIS = ((60, 60, 255), (60, 255, 60), (255, 120, 60))
COLOR_TEXT = (255, 255, 255)
COLOR_TEXT_SHADOW = (0, 0, 0)

#: Cube wireframe edges in canonical vertex order (see cube_pose_3d).
CUBE_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)
CANONICAL_VERTEX_SIGNS = (
    (-1, -1, -1), (+1, -1, -1), (+1, +1, -1), (-1, +1, -1),
    (-1, -1, +1), (+1, -1, +1), (+1, +1, +1), (-1, +1, +1),
)


class OverlayError(ValueError):
    """Raised when the overlay cannot be drawn from the given evidence."""


def _put(cv2, image, text, origin, *, color=COLOR_TEXT, scale=0.5, thickness=1):
    """Draw text with a dark outline so it survives any background."""
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
                COLOR_TEXT_SHADOW, thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def _project(point_object, position_m, orientation_xyzw, camera_matrix):
    """Project an object-frame point through the pose and pinhole model."""
    x, y, z, w = (float(value) for value in orientation_xyzw)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm < 1e-12:
        return None
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    ux, uy, uz = x, y, z
    vx, vy, vz = (float(value) for value in point_object)
    cross = (uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx)
    dot = ux * vx + uy * vy + uz * vz
    uu = ux * ux + uy * uy + uz * uz
    rotated = tuple(
        2.0 * dot * u + (w * w - uu) * v + 2.0 * w * c
        for u, v, c in zip((ux, uy, uz), (vx, vy, vz), cross)
    )
    camera = tuple(rotated[i] + float(position_m[i]) for i in range(3))
    if camera[2] <= 1e-6:
        return None
    k = [float(value) for value in camera_matrix]
    u = k[0] * camera[0] / camera[2] + k[1] * camera[1] / camera[2] + k[2]
    v = k[3] * camera[0] / camera[2] + k[4] * camera[1] / camera[2] + k[5]
    return (float(u), float(v))


def draw_cube_overlay(
    image,
    *,
    camera_matrix=None,
    pose=None,
    bbox_xyxy=None,
    bbox_source='',
    bbox_quality=None,
    silhouette_corners=None,
    face_regions=None,
    observed_colors=None,
    visible_faces=None,
    status='unknown',
    reason='',
    reprojection_error_px=None,
    color_resolved=None,
    label_swaps=None,
    side_length_m=0.057,
    lines=None,
):
    """Return a copy of ``image`` annotated with the perception evidence.

    Everything is optional: whatever the caller has is drawn, whatever is
    missing is simply absent. ``status`` drives the header colour so a rejected
    frame is unmistakable at a glance.
    """
    import cv2

    if image is None or getattr(image, 'ndim', 0) != 3:
        raise OverlayError('overlay needs an HxWx3 BGR image')
    canvas = np.array(image, dtype=np.uint8, copy=True)
    height, width = canvas.shape[:2]

    header_color = {
        'accepted': COLOR_ACCEPTED,
        'resolved': COLOR_ACCEPTED,
        'rejected': COLOR_REJECTED,
        'unresolved': COLOR_PENDING,
    }.get(str(status).lower(), COLOR_PENDING)

    # 1. the box that was used, and where it came from
    if bbox_xyxy is not None and len(bbox_xyxy) == 4:
        left, top, right, bottom = (int(round(float(v))) for v in bbox_xyxy)
        cv2.rectangle(canvas, (left, top), (right, bottom), COLOR_BOX, 1)
        label = f'bbox: {bbox_source or "detector"}'
        if bbox_quality is not None:
            label += f' q={float(bbox_quality):.2f}'
        _put(cv2, canvas, label, (max(4, left), max(16, top - 6)),
             color=COLOR_BOX, scale=0.45)

    # 2. the silhouette the corners were snapped to
    if silhouette_corners:
        polygon = np.array(
            [[int(round(float(p[0]))), int(round(float(p[1])))]
             for p in silhouette_corners], dtype=np.int32
        )
        if polygon.shape[0] >= 2:
            cv2.polylines(canvas, [polygon], True, COLOR_SILHOUETTE, 1)

    # 3. the visible face regions with the colour they were classified as
    if face_regions:
        for slot, quad in face_regions.items():
            polygon = None
            vertices = getattr(quad, 'vertices', quad)
            try:
                polygon = np.array(
                    [[int(round(float(p[0]))), int(round(float(p[1])))]
                     for p in vertices], dtype=np.int32
                )
            except (TypeError, ValueError, IndexError):
                continue
            if polygon.shape[0] < 3:
                continue
            cv2.polylines(canvas, [polygon], True, COLOR_FACE, 2)
            label = slot
            if visible_faces and slot in visible_faces and observed_colors:
                label += '=' + str(observed_colors.get(slot) or '?')
            anchor = polygon[0]
            _put(cv2, canvas, label,
                 (int(anchor[0]) + 4, int(anchor[1]) - 6),
                 color=COLOR_FACE, scale=0.5)

    # 4. the pose as a wireframe plus axes: a wrong pose is visible here
    if pose is not None and camera_matrix is not None:
        position = pose.get('position_m')
        orientation = pose.get('orientation_xyzw')
        if position and orientation and len(position) == 3:
            half = float(side_length_m) / 2.0
            projected = {}
            for index, signs in enumerate(CANONICAL_VERTEX_SIGNS):
                point = tuple(sign * half for sign in signs)
                pixel = _project(point, position, orientation, camera_matrix)
                if pixel is not None:
                    projected[index] = (int(round(pixel[0])),
                                        int(round(pixel[1])))
            for first, second in CUBE_EDGES:
                if first in projected and second in projected:
                    cv2.line(canvas, projected[first], projected[second],
                             COLOR_ACCEPTED, 2, cv2.LINE_AA)
            origin = _project((0.0, 0.0, 0.0), position, orientation,
                              camera_matrix)
            if origin is not None:
                for axis, color in zip(
                        ((1, 0, 0), (0, 1, 0), (0, 0, 1)), COLOR_AXIS):
                    tip = _project(
                        tuple(component * half * 1.8 for component in axis),
                        position, orientation, camera_matrix,
                    )
                    if tip is None:
                        continue
                    cv2.line(canvas,
                             (int(round(origin[0])), int(round(origin[1]))),
                             (int(round(tip[0])), int(round(tip[1]))),
                             color, 2, cv2.LINE_AA)
            for point in projected.values():
                cv2.circle(canvas, point, 3, COLOR_ACCEPTED, -1)

    # 5. the status line: the answer to "why is there no pose"
    _put(cv2, canvas, f'{status}  {reason}'.strip(),
         (10, 22), color=header_color, scale=0.55, thickness=2)
    details = []
    if reprojection_error_px is not None:
        details.append(f'rms={float(reprojection_error_px):.2f}px')
    if bbox_source:
        details.append(f'bbox={bbox_source}')
    if color_resolved is not None:
        details.append(
            'colors=resolved' if color_resolved else 'colors=unresolved'
        )
    if label_swaps is not None:
        details.append('swaps=' + ''.join(
            'T' if value else 'F' for value in label_swaps
        ))
    if details:
        _put(cv2, canvas, '  '.join(details), (10, 44),
             color=COLOR_TEXT, scale=0.45)

    if observed_colors:
        rendered = ', '.join(
            f'{slot}:{label or "?"}' for slot, label in
            sorted(observed_colors.items())
        )
        _put(cv2, canvas, rendered, (10, height - 12),
             color=COLOR_FACE, scale=0.45)

    if lines:
        for index, text in enumerate(lines):
            _put(cv2, canvas, str(text), (10, 66 + 18 * index),
                 color=COLOR_TEXT, scale=0.45)

    return canvas


def overlay_is_blank(image):
    """Report whether nothing was drawn: a flat frame with a single colour."""
    array = np.asarray(image)
    if array.size == 0:
        return True
    return int(np.unique(array.reshape(-1, array.shape[2] if array.ndim == 3
                                       else 1), axis=0).shape[0]) <= 1
