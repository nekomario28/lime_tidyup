"""Refine a cube pose from its silhouette, given a prior.

Recognition only: no ROS, no dynamics. This is the middle ground the survey
called for -- a single frame at range has too few pixels for the two-face solve,
but with a prior (the last good pose) the cube's silhouette is still measurable,
and that is enough for PnP.

The pipeline is deliberately small and reuses what already exists:

1. build a colour mask with the same rule as the bbox source (saturated hue on a
   dark floor), closing over the sticker grooves;
2. take the silhouette's corners from that mask;
3. project the eight canonical vertices of the prior pose and keep the visible
   ones (``cube_visibility.visible_vertex_mask``);
4. match each visible vertex to the nearest observed corner, and only count a
   match within ``maximum_corner_distance_px``;
5. hand those correspondences to ``estimate_cube_pose_3d``, which validates the
   reprojection like every other pose in this package.

A prior that is far off produces few or no matches, and the function raises
instead of returning a confident-looking wrong pose.

An axis-aligned cube viewed frontally presents four coplanar silhouette corners,
and PnP on coplanar points has a mirror ambiguity: measured on real sim frames at
0.4-1.0 m the solve either returns a pose flipped by about 130 degrees or is
refused by the reprojection gate. A general orientation (or a view high enough to
show the top face's far corners, i.e. six non-coplanar vertices) solves cleanly.
So a refusal here is evidence about the geometry, not a bug -- and it is why the
two-face route in this package needs roughly 250 px of cube before it will answer.
"""
from __future__ import annotations

import math

import numpy as np

from .cube_pose_3d import CubePose3DError, estimate_cube_pose_3d
from .cube_visibility import CANONICAL_VERTEX_SIGNS, visible_vertex_mask

#: Same hue centres and gates as cube_bbox_source.colour_blob_bbox.
HUE_CENTRES = (('red', 3.0), ('orange', 16.0), ('yellow', 30.0),
               ('green', 60.0), ('blue', 110.0))
HUE_TOLERANCE = 12.0
MINIMUM_SATURATION = 90
MINIMUM_VALUE = 40
CLOSE_KERNEL = (11, 11)


class CubePoseTrackingError(ValueError):
    """Raised when the prior and the silhouette cannot be reconciled."""


def colour_mask(image_bgr):
    """Return (mask, area) for the most distinctly coloured region."""
    import cv2

    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[2] != 3:
        raise CubePoseTrackingError('expected an HxWx3 BGR image')
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.float32)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    best_mask = None
    best_area = 0
    for _label, centre in HUE_CENTRES:
        distance = np.minimum(np.abs(hue - centre), 180.0 - np.abs(hue - centre))
        mask = ((saturation >= MINIMUM_SATURATION) & (value >= MINIMUM_VALUE)
                & (distance <= HUE_TOLERANCE)).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                np.ones(CLOSE_KERNEL, np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if count < 2:
            continue
        index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area > best_area:
            best_area = area
            best_mask = (labels == index).astype(np.uint8) * 255
    if best_mask is None:
        raise CubePoseTrackingError('no distinct colour region in the frame')
    return best_mask, best_area


#: A pixel belongs to the object when its Lab colour differs from the surface
#: behind it by more than this. The hue mask alone only covers the saturated
#: faces: measured on real frames it returned the red front face as a quad,
#: which is not the cube's silhouette and made every correspondence wrong.
OBJECT_BACKGROUND_DISTANCE = 25.0
BACKGROUND_ROWS_FRACTION = 0.10


def object_mask(image_bgr, seed_mask, *, distance_threshold=OBJECT_BACKGROUND_DISTANCE):
    """Full object mask: everything that differs from the surface behind it.

    The surface colour is taken from the bottom rows of the frame, and the largest
    region near the seed (the saturated blob) is kept, so unrelated bright areas
    elsewhere in the image cannot join the silhouette.
    """
    import cv2

    image = np.asarray(image_bgr)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    rows = max(1, int(lab.shape[0] * BACKGROUND_ROWS_FRACTION))
    background = np.median(lab[-rows:, :, :].reshape(-1, 3), axis=0)
    distance = np.linalg.norm(lab - background, axis=2)
    mask = (distance > float(distance_threshold)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones(CLOSE_KERNEL, np.uint8))

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    if count < 2:
        raise CubePoseTrackingError('object mask is empty')
    seed_points = np.argwhere(seed_mask > 0)
    if seed_points.size == 0:
        raise CubePoseTrackingError('seed mask is empty')
    seed_centre = seed_points.mean(axis=0)
    best = None
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] < 50:
            continue
        centre = centroids[index]
        gap = math.dist((centre[1], centre[0]),
                        (float(seed_centre[1]), float(seed_centre[0])))
        if best is None or gap < best[0]:
            best = (gap, index)
    if best is None:
        raise CubePoseTrackingError('no candidate region near the seed')
    region = (labels == best[1]).astype(np.uint8) * 255
    return region, int(stats[best[1], cv2.CC_STAT_AREA])


def silhouette_corners(mask, *, maximum_corners=8, epsilon_fraction=0.01):
    """Corners of the largest silhouette, plus its contour, in pixels."""
    import cv2

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise CubePoseTrackingError('mask has no contour')
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0.0:
        raise CubePoseTrackingError('silhouette has no perimeter')
    polygon = cv2.approxPolyDP(contour, epsilon_fraction * perimeter, True)
    points = polygon.reshape(-1, 2).astype(np.float64)
    if len(points) > maximum_corners:
        # keep the strongest corners: the ones furthest from their neighbours
        keep = []
        for index in range(len(points)):
            previous = points[index - 1]
            nxt = points[(index + 1) % len(points)]
            span = math.dist(previous, points[index]) + math.dist(points[index], nxt)
            keep.append((span, index))
        keep.sort(reverse=True)
        points = np.asarray([points[index] for _span, index in keep[:maximum_corners]])
    if len(points) < 3:
        raise CubePoseTrackingError('silhouette has fewer than three corners')
    return points, contour


def vertices_object_frame(side_length_m):
    half = 0.5 * float(side_length_m)
    return np.asarray([[sign * half for sign in signs]
                       for signs in CANONICAL_VERTEX_SIGNS], dtype=np.float64)


def quaternion_matrix(orientation_xyzw):
    x, y, z, w = (float(value) for value in orientation_xyzw)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0.0:
        raise CubePoseTrackingError('orientation has zero norm')
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def project(points_camera, camera_matrix):
    k = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    points = np.asarray(points_camera, dtype=np.float64).reshape(-1, 3)
    if np.any(points[:, 2] <= 0.0):
        raise CubePoseTrackingError('a vertex is behind the camera')
    pixels = points @ k.T
    return pixels[:, :2] / pixels[:, 2:3]


def _cyclic_order(points):
    """Indices of 2-D points in cyclic (counter-clockwise) order about their centre."""
    centre = np.mean(points, axis=0)
    angles = [math.atan2(point[1] - centre[1], point[0] - centre[0])
              for point in points]
    return [index for index, _angle in sorted(enumerate(angles),
                                              key=lambda item: item[1])]


def _hull_indices(points):
    """Indices of the points that lie on their own convex hull."""
    import cv2

    hull = cv2.convexHull(points.astype(np.float32), returnPoints=False)
    return set(int(index) for index in hull.reshape(-1))


def refine_pose_from_silhouette(
    image_bgr,
    camera_matrix,
    prior_position_m,
    prior_orientation_xyzw,
    *,
    side_length_m=0.057,
    maximum_corner_distance_px=15.0,
    maximum_edge_distance_px=3.0,
    minimum_matches=4,
    maximum_reprojection_error_px=4.0,
    minimum_keypoint_confidence=0.5,
):
    """Return (pose, evidence) refined from the silhouette around the prior."""
    import cv2

    seed, _seed_area = colour_mask(image_bgr)
    mask, area = object_mask(image_bgr, seed)
    corners, contour = silhouette_corners(mask)
    rotation = quaternion_matrix(prior_orientation_xyzw)
    object_points = vertices_object_frame(side_length_m)
    camera_points = object_points @ rotation.T + np.asarray(prior_position_m,
                                                            dtype=np.float64)
    predicted = project(camera_points, camera_matrix)
    visible = visible_vertex_mask(prior_position_m, prior_orientation_xyzw,
                                  side_length_m=side_length_m)

    # Cyclic correspondence instead of nearest-neighbour: the silhouette is the
    # cube's hull, so its corners and the prior's hull vertices are the same cycle
    # up to a rotation and a direction. Nearest-neighbour pairs a vertex with its
    # neighbour's corner as soon as the prior is off, and no pose fits that
    # (measured: PnP rejected every subset). Order both by angle and try every
    # shift, keeping the pose with the smallest reprojection error.
    predicted_hull = _hull_indices(predicted)
    visible_hull = [index for index in range(len(object_points))
                    if visible[index] and index in predicted_hull]
    if len(visible_hull) < minimum_matches or len(corners) < minimum_matches:
        raise CubePoseTrackingError(
            'not enough hull vertices: %d predicted, %d observed'
            % (len(visible_hull), len(corners)))

    # Angle-based one-to-one pairing about the centroid. The observed silhouette
    # often has fewer corners than the prior predicts (a 40 px cube at 1 m gives
    # four), so a cyclic shift with equal counts drops the wrong pairs; the
    # direction from the centre is stable regardless of how many corners survive.
    predicted_centre = np.mean(predicted[visible_hull], axis=0)
    observed_centre = np.mean(corners, axis=0)
    predicted_angles = [math.atan2(predicted[index][1] - predicted_centre[1],
                                   predicted[index][0] - predicted_centre[0])
                        for index in visible_hull]
    observed_angles = [math.atan2(corner[1] - observed_centre[1],
                                  corner[0] - observed_centre[0])
                       for corner in corners]

    pairs = []
    for corner_index, observed_angle in enumerate(observed_angles):
        best = None
        for slot, predicted_angle in enumerate(predicted_angles):
            difference = abs(math.atan2(math.sin(predicted_angle - observed_angle),
                                        math.cos(predicted_angle - observed_angle)))
            if best is None or difference < best[0]:
                best = (difference, slot)
        pairs.append((best[0], visible_hull[best[1]], corner_index))

    points = np.zeros((len(object_points), 2), dtype=np.float64)
    weights = np.zeros(len(object_points), dtype=np.float64)
    matches = []
    used_vertices = set()
    used_corners = set()
    for _difference, vertex_index, corner_index in sorted(pairs):
        if vertex_index in used_vertices or corner_index in used_corners:
            continue
        used_vertices.add(vertex_index)
        used_corners.add(corner_index)
        points[vertex_index] = corners[corner_index]
        weights[vertex_index] = 1.0
        matches.append((vertex_index, corner_index))

    pose = None
    try:
        pose = estimate_cube_pose_3d(
            [tuple(point) for point in points],
            [float(value) for value in weights],
            camera_matrix,
            side_length_m=float(side_length_m),
            minimum_keypoint_confidence=float(minimum_keypoint_confidence),
            minimum_inliers=int(minimum_matches),
            maximum_reprojection_error_px=float(maximum_reprojection_error_px),
        )
    except CubePose3DError as error:
        raise CubePoseTrackingError(
            'pnp rejected the refinement: %s' % error) from error
    if len(matches) < minimum_matches:
        raise CubePoseTrackingError(
            'only %d corners could be paired with a hull vertex (need %d)'
            % (len(matches), minimum_matches))

    evidence = {
        'silhouette_area_px': int(area),
        'silhouette_corners': int(len(corners)),
        'visible_vertices': int(sum(visible)),
        'matched_vertices': len(matches),
    }
    return pose, evidence
