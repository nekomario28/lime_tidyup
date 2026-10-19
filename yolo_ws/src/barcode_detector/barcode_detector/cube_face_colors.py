"""Face colour evidence: sample the visible faces and label them, or abstain.

The pose hypotheses in :mod:`cube_pose_hypotheses` are separated only by the
colours of the visible faces, so the colour leg has to be as strict as the
geometry leg: a label is reported only when one reference colour is clearly
closer than the next, otherwise the face is ``None`` (unknown) and the caller
must keep the pose unresolved. Inventing a third colour or snapping a grey face
to the nearest hue is exactly the failure this module exists to avoid.

Sampling uses the sticker centres of a face (a 3x3 grid inside the quad), not
the whole quad, so black separators, silhouette edges and glare at the corners
cannot dominate the estimate. Pixel values are converted to CIELAB and compared
on the chroma plane with lightness down-weighted, because a shaded face is a
darker version of the same colour, not a different one.
"""

from __future__ import annotations

import math

import numpy as np

from .cube_pose_hypotheses import FACE_NORMALS

#: Standard six-colour cube palette in sRGB, used as anchors.
REFERENCE_RGB = {
    'white': (245, 245, 245),
    'yellow': (255, 213, 0),
    'red': (183, 18, 52),
    'orange': (255, 88, 0),
    'green': (0, 155, 72),
    'blue': (0, 70, 173),
}

#: Below this chroma a face carries no hue evidence (white is handled by L).
MINIMUM_CHROMA = 10.0
#: Largest circular hue distance, in degrees, that still counts as a match.
MAXIMUM_HUE_DISTANCE_DEG = 20.0
#: Required hue gap, in degrees, between the best and second-best anchor.
MINIMUM_HUE_MARGIN_DEG = 8.0
#: Accepted chroma ratio against the anchor; shading changes L, not hue.
MINIMUM_CHROMA_RATIO = 0.2
MAXIMUM_CHROMA_RATIO = 5.0
#: A face is sampled at its nine sticker centres.
STICKER_OFFSETS = (-2.0 / 3.0, 0.0, 2.0 / 3.0)
#: Half-width, in pixels, of the patch averaged at each sticker centre.
PATCH_RADIUS_PX = 1


def _linearize(channel):
    value = channel / 255.0
    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


def rgb_to_lab(rgb):
    """Convert an sRGB triple (0-255) to CIELAB with a D65 white point."""
    red, green, blue = (_linearize(float(value)) for value in rgb)
    x = red * 0.4124 + green * 0.3576 + blue * 0.1805
    y = red * 0.2126 + green * 0.7152 + blue * 0.0722
    z = red * 0.0193 + green * 0.1192 + blue * 0.9505
    xn, yn, zn = 0.95047, 1.0, 1.08883

    def transform(value):
        if value > 0.008856:
            return value ** (1.0 / 3.0)
        return 7.787 * value + 16.0 / 116.0

    fx, fy, fz = transform(x / xn), transform(y / yn), transform(z / zn)
    return (116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz))


REFERENCE_LAB = {
    label: rgb_to_lab(rgb) for label, rgb in REFERENCE_RGB.items()
}


def _hue_deg(a, b):
    return math.degrees(math.atan2(b, a))


def _hue_distance_deg(first, second):
    difference = abs(first - second) % 360.0
    return min(difference, 360.0 - difference)


def classify_lab(
    lab,
    *,
    minimum_chroma=MINIMUM_CHROMA,
    maximum_hue_distance_deg=MAXIMUM_HUE_DISTANCE_DEG,
    minimum_hue_margin_deg=MINIMUM_HUE_MARGIN_DEG,
    minimum_chroma_ratio=MINIMUM_CHROMA_RATIO,
    maximum_chroma_ratio=MAXIMUM_CHROMA_RATIO,
):
    """Return the reference label for ``lab``, or ``None`` if unclear.

    Hue carries the identity and chroma magnitude only has to stay in range, so
    a shaded face keeps its label while a desaturated one abstains. ``None``
    covers three cases on purpose: an achromatic patch too dark to be white, a
    patch whose hue sits between two anchors, and a patch whose chroma is far
    from every anchor. All three mean "do not claim a colour", which is what
    the caller needs to keep the pose unresolved.
    """
    lightness, a, b = (float(value) for value in lab)
    if not all(math.isfinite(value) for value in (lightness, a, b)):
        return None
    chroma = math.hypot(a, b)
    if chroma < float(minimum_chroma):
        return 'white' if lightness > 80.0 else None

    hue = _hue_deg(a, b)
    candidates = []
    for label, (_, ref_a, ref_b) in REFERENCE_LAB.items():
        reference_chroma = math.hypot(ref_a, ref_b)
        if reference_chroma < float(minimum_chroma):
            continue  # achromatic anchors are decided by lightness alone
        ratio = chroma / reference_chroma
        if not (float(minimum_chroma_ratio) <= ratio
                <= float(maximum_chroma_ratio)):
            continue
        candidates.append(
            (_hue_distance_deg(hue, _hue_deg(ref_a, ref_b)), label)
        )
    if not candidates:
        return None
    candidates.sort()
    best_distance, best_label = candidates[0]
    if best_distance > float(maximum_hue_distance_deg):
        return None
    if len(candidates) > 1:
        second_distance = candidates[1][0]
        if second_distance - best_distance < float(minimum_hue_margin_deg):
            return None
    return best_label


def _face_corner_object_points(face_name, side_length_m):
    half = float(side_length_m) / 2.0
    normal = FACE_NORMALS[face_name]
    axis = [index for index in range(3) if abs(normal[index]) > 0.5][0]
    others = [index for index in range(3) if index != axis]
    corners = []
    for first, second in ((-1, -1), (-1, 1), (1, 1), (1, -1)):
        point = [0.0, 0.0, 0.0]
        point[axis] = normal[axis] * half
        point[others[0]] = first * half
        point[others[1]] = second * half
        corners.append(tuple(point))
    return corners, (others[0], others[1])


def _rotate(quaternion, vector):
    x, y, z, w = (float(value) for value in quaternion)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError('invalid orientation quaternion')
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    ux, uy, uz = x, y, z
    vx, vy, vz = vector
    cross = (uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx)
    dot = ux * vx + uy * vy + uz * vz
    uu = ux * ux + uy * uy + uz * uz
    return tuple(
        2.0 * dot * u + (w * w - uu) * v + 2.0 * w * c
        for u, v, c in zip((ux, uy, uz), (vx, vy, vz), cross)
    )


def project_point(point_object, position_m, orientation_xyzw, camera_matrix):
    """Project an object-frame point through the pose and the pinhole model."""
    rotated = _rotate(orientation_xyzw, point_object)
    camera = tuple(rotated[i] + float(position_m[i]) for i in range(3))
    if camera[2] <= 1e-6:
        return None
    k = [float(value) for value in camera_matrix]
    u = k[0] * camera[0] / camera[2] + k[1] * camera[1] / camera[2] + k[2]
    v = k[3] * camera[0] / camera[2] + k[4] * camera[1] / camera[2] + k[5]
    return (u, v)


def _sticker_centres(face_name, side_length_m):
    corners, axes = _face_corner_object_points(face_name, side_length_m)
    half = float(side_length_m) / 2.0
    centres = []
    for first in STICKER_OFFSETS:
        for second in STICKER_OFFSETS:
            point = [0.0, 0.0, 0.0]
            normal = FACE_NORMALS[face_name]
            axis = [index for index in range(3)
                    if abs(normal[index]) > 0.5][0]
            point[axis] = normal[axis] * half
            point[axes[0]] = first * half
            point[axes[1]] = second * half
            centres.append(tuple(point))
    del corners
    return centres


def _face_outline(face_name, position_m, orientation_xyzw, camera_matrix,
                  side_length_m):
    """Projected corners of a face, or ``None`` if any corner is behind."""
    corners, _ = _face_corner_object_points(face_name, side_length_m)
    outline = []
    for corner in corners:
        pixel = project_point(
            corner, position_m, orientation_xyzw, camera_matrix
        )
        if pixel is None:
            return None
        outline.append([round(pixel[0], 1), round(pixel[1], 1)])
    return outline


def _patch_lab(image, u, v, radius):
    height, width = image.shape[0], image.shape[1]
    x = int(round(u))
    y = int(round(v))
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    patch = image[y0:y1, x0:x1]
    blue, green, red = (
        float(np.median(patch[:, :, channel])) for channel in range(3)
    )
    return rgb_to_lab((red, green, blue))


def observe_face_colors(
    image,
    position_m,
    orientation_xyzw,
    camera_matrix,
    visible_slots,
    *,
    side_length_m,
    minimum_chroma=MINIMUM_CHROMA,
    maximum_hue_distance_deg=MAXIMUM_HUE_DISTANCE_DEG,
    minimum_hue_margin_deg=MINIMUM_HUE_MARGIN_DEG,
    patch_radius_px=PATCH_RADIUS_PX
):
    """Return the observed colour of each visible face, or ``None`` per face.

    A face with too few sampled pixels, or whose colour does not clearly match
    one anchor, yields ``None``. The returned evidence is kept alongside the
    labels so a caller can log why a face abstained.
    """
    observed = {}
    evidence = {}
    for slot in visible_slots:
        points = []
        for centre in _sticker_centres(slot, side_length_m):
            projected = project_point(
                centre, position_m, orientation_xyzw, camera_matrix
            )
            if projected is None:
                continue
            lab = _patch_lab(image, projected[0], projected[1],
                             patch_radius_px)
            if lab is not None:
                points.append(lab)
        if len(points) < 4:
            observed[slot] = None
            evidence[slot] = {'status': 'insufficient_samples',
                              'sample_count': len(points)}
            continue
        median = tuple(
            float(np.median([point[index] for point in points]))
            for index in range(3)
        )
        label = classify_lab(
            median,
            minimum_chroma=minimum_chroma,
            maximum_hue_distance_deg=maximum_hue_distance_deg,
            minimum_hue_margin_deg=minimum_hue_margin_deg,
        )
        observed[slot] = label
        evidence[slot] = {
            'status': 'observed' if label is not None else 'unknown',
            'sample_count': len(points),
            'lab': [round(value, 3) for value in median],
            'chroma': round(math.hypot(median[1], median[2]), 3),
            # the face outline, so a viewer can show where the colour came from
            'quad': _face_outline(
                slot, position_m, orientation_xyzw, camera_matrix,
                side_length_m,
            ),
        }
    return observed, evidence


def face_color_labels():
    """The anchor labels a caller may see, in deterministic order."""
    return tuple(sorted(REFERENCE_LAB))
