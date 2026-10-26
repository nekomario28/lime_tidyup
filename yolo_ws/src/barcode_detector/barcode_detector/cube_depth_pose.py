"""Cube orientation and centre from an aligned depth image.

A single-coloured cube gives the colour frontends nothing to resolve: the
two-face route needs two differently coloured faces, and the trained model here
carries seven keypoints, not the eight ordered corners ``cube_pose_3d`` wants.
Depth does not care about colour.

The cube stands on the floor, so two measured planes are enough to build its
frame: the surface below the box gives the up axis, and the box's own dominant
vertical face gives one horizontal axis. The known side length then turns the
visible face into the centre.

Directions come from *pixel normals*, not from a least-squares plane fit. A fit
over the sparse valid points below a box is ill-conditioned -- measured on the
robot 2026-10-26 it returned a plane 24 degrees off the tilt of the same
surface's normals -- while the normals of a flat desk are consistent whatever
the density.

What this route does *not* claim:

* It is not a pose of labelled faces. A cube is symmetric under a quarter turn
  about any axis, so the frame returned is one of four equivalent ones. A
  consumer that needs the jaws across a pair of faces snaps the tool yaw to a
  quarter turn (``cm1``'s ``arm_to_cube`` does).
* It needs a visible vertical face. A cube seen from directly above has no
  horizontal axis to measure, and this route withholds rather than guesses.

The up axis comes from ``up_hint_xyz`` when the caller has one -- the robot's own
``base_link`` Z expressed in the camera frame is exact, and it is the frame the
arm moves in. Without a hint the floor below the box supplies it, which is
weaker: measured on the robot 2026-10-26, the region under a box held both the
floor and a fronto-parallel wall panel, and the pose was refused on half the
frames because the panel won the vote.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

GEOMETRY_ID = 'depth-face-axis-v1'

#: In an optical frame the floor normal points away from the row axis (y down),
#: so the sign of the floor direction is fixed against this: "up" then means away
#: from the floor, whichever way the camera is tilted.
_OPTICAL_DOWN = np.array([0.0, 1.0, 0.0])

#: Depth samples a depth image can be trusted for, in metres.
_MINIMUM_DEPTH_M = 0.02
_MAXIMUM_DEPTH_M = 10.0


class CubeDepthPoseError(ValueError):
    """Raised when the depth image cannot support a pose."""


@dataclass(frozen=True)
class CubeDepthPose:
    """Camera-frame cube centre and frame, with the evidence that produced it."""

    position_m: tuple
    orientation_xyzw: tuple
    up_xyz: tuple
    up_source: str
    up_share: float
    up_tilt_deg: float
    face_share: float
    face_support: int
    face_residual_m: float
    in_plane_extent_m: float
    side_length_m: float

    def as_dict(self, *, frame_id, timestamp):
        """Return the ``/cube_pose_result`` pose payload."""
        return {
            'geometry_id': GEOMETRY_ID,
            'frame_id': frame_id,
            'timestamp': float(timestamp),
            'side_length_m': float(self.side_length_m),
            'position_m': list(self.position_m),
            'orientation_xyzw': list(self.orientation_xyzw),
            'reprojection_error_px': None,
            'correspondence_count': int(self.face_support),
            'inlier_count': int(self.face_support),
            'inlier_mask': [],
            'color_fixed_semantics_resolved': False,
            'up_xyz': list(self.up_xyz),
            'up_source': self.up_source,
            'up_share': self.up_share,
            'up_tilt_deg': self.up_tilt_deg,
            'face_share': self.face_share,
            'face_residual_m': self.face_residual_m,
            'in_plane_extent_m': self.in_plane_extent_m,
            'perception_backend': 'depth_geometry',
        }


def _unit(vector):
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm < 1e-9:
        raise CubeDepthPoseError('degenerate direction')
    return vector / norm


def _camera_matrix(matrix):
    values = np.asarray([float(value) for value in matrix], dtype=np.float64)
    if values.size != 9:
        raise CubeDepthPoseError('camera matrix K must contain 9 values')
    k = values.reshape(3, 3)
    if not (k[0, 0] > 0.0 and k[1, 1] > 0.0):
        raise CubeDepthPoseError('camera matrix has no focal length')
    return k


def _valid_mask(window):
    """Samples a depth image can be trusted for, in metres."""
    return (np.isfinite(window) & (window > _MINIMUM_DEPTH_M)
            & (window < _MAXIMUM_DEPTH_M))


def _points(window, valid, k, u0, v0):
    """Back-project the valid samples of a window into camera-frame points."""
    v_index, u_index = np.nonzero(valid)
    z = window[valid].astype(np.float64)
    u = (u_index + u0).astype(np.float64)
    v = (v_index + v0).astype(np.float64)
    x = (u - k[0, 2]) * z / k[0, 0]
    y = (v - k[1, 2]) * z / k[1, 1]
    return np.stack([x, y, z], axis=1)


def _window_points(depth, k, u0, u1, v0, v1, minimum_samples):
    """Points and their pixel normals for a region given in image coordinates."""
    v0 = max(0, min(depth.shape[0] - 1, int(v0)))
    v1 = max(v0 + 1, min(depth.shape[0], int(v1)))
    u0 = max(0, min(depth.shape[1] - 1, int(u0)))
    u1 = max(u0 + 1, min(depth.shape[1], int(u1)))
    window = depth[v0:v1, u0:u1]
    valid = _valid_mask(window)
    if valid.sum() < minimum_samples:
        raise CubeDepthPoseError(
            'only %d valid depth samples in the region' % int(valid.sum()))
    points = _points(window, valid, k, u0, v0)
    normals = _pixel_normals(points, valid, window.shape)
    return points, normals


def _box_windows(depth, box_xyxy, roi_fraction):
    """The box's interior (for directions) and the whole box (for the centre).

    Near a box edge a pixel normal mixes the cube with the floor behind it, so
    directions come from an inset window. The centre needs the whole box: the
    inset clips the face, and that clipping lands in the in-plane midpoints.
    """
    x0, y0, x1, y1 = (float(value) for value in box_xyxy)
    inset_x = float(roi_fraction) * (x1 - x0)
    inset_y = float(roi_fraction) * (y1 - y0)
    inner = (math.floor(x0 + inset_x), math.ceil(x1 - inset_x),
             math.floor(y0 + inset_y), math.ceil(y1 - inset_y))
    if inner[1] - inner[0] < 4 or inner[3] - inner[2] < 4:
        raise CubeDepthPoseError('box is too small to sample')
    whole = (math.floor(x0), math.ceil(x1), math.floor(y0), math.ceil(y1))
    return inner, whole


def _pixel_normals(points, valid, shape):
    """Normals from neighbouring back-projected points, in the optical frame.

    A sample whose neighbours are missing gets NaN and is dropped by the caller:
    a normal computed across a depth hole points nowhere in particular.
    """
    rows, cols = shape
    point_map = np.full((rows, cols, 3), np.nan)
    point_map[valid] = points
    across = point_map[1:-1, 2:] - point_map[1:-1, :-2]
    down = point_map[2:, 1:-1] - point_map[:-2, 1:-1]
    with np.errstate(invalid='ignore'):
        cross = np.cross(across, down)
        normals = cross / np.linalg.norm(cross, axis=2, keepdims=True)
    normal_map = np.full((rows, cols, 3), np.nan)
    normal_map[1:-1, 1:-1] = normals
    return normal_map[valid]


def _dominant_direction(normals, sample=400, tolerance_deg=12.0):
    """The direction most of the normals agree with, and how many agreed.

    A plane's pixels all point the same way, so the mode is the surface. The
    comparison folds the sign: a face and its back are the same plane.
    """
    usable = normals[np.isfinite(normals).all(axis=1)]
    if usable.shape[0] < 20:
        raise CubeDepthPoseError('too few pixel normals to find a surface')
    step = max(1, usable.shape[0] // int(sample))
    sample_rows = usable[::step]
    limit = math.cos(math.radians(float(tolerance_deg)))
    best_seed, best_group = None, None
    for seed in sample_rows:
        group = np.abs(sample_rows @ seed) >= limit
        if best_group is None or group.sum() > best_group.sum():
            best_seed, best_group = seed, group
    # The comparison folded the sign, so the members can disagree about it: align
    # them before averaging, or a mixed group cancels to nothing and the "mode"
    # comes out as some unrelated direction.
    members = sample_rows[best_group]
    aligned = members * np.sign(members @ best_seed)[:, None]
    return _unit(aligned.mean(axis=0)), int(best_group.sum()), sample_rows.shape[0]


def _dense_band(values, bin_m=0.002, minimum_share=0.25, minimum_count=100):
    """The centre of the densest band of values: a plane perpendicular to an axis.

    Only a plane that faces the axis forms such a band; every other surface in a
    box around a cube spreads out along it -- including the floor in front of the
    cube, which reaches further along a tilted axis than the face does.

    The band is a *run* of adjacent occupied bins, not one bin. An axis a couple
    of degrees off the face normal spreads that face over several bins (measured
    on the robot 2026-10-26: 6 mm of spread), and picking the furthest occupied
    bin then lands on the sparse tail of the face instead of the face.
    """
    edges = np.arange(values.min(), values.max() + bin_m, bin_m)
    if edges.size < 2:
        raise CubeDepthPoseError('nothing to fit a face plane to')
    counts, edges = np.histogram(values, bins=edges)
    occupied = counts >= max(minimum_count, int(minimum_share * counts.max()))
    runs = []
    start = None
    for index, flag in enumerate(occupied):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, occupied.size - 1))
    if not runs:
        raise CubeDepthPoseError('no vertical face found in the box')
    first, last = max(runs, key=lambda run: sum(counts[run[0]:run[1] + 1]))
    centres = [0.5 * (edges[index] + edges[index + 1])
               for index in range(first, last + 1)]
    weights = [float(counts[index]) for index in range(first, last + 1)]
    return float(np.average(centres, weights=weights))


def _quaternion_from_axes(x_axis, y_axis, z_axis):
    """The XYZW quaternion of the rotation whose columns are the three axes."""
    matrix = np.stack([x_axis, y_axis, z_axis], axis=1)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        x = (matrix[2, 1] - matrix[1, 2]) / s
        y = (matrix[0, 2] - matrix[2, 0]) / s
        z = (matrix[1, 0] - matrix[0, 1]) / s
        w = 0.25 * s
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            s = math.sqrt(
                1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            x, w = 0.25 * s, (matrix[2, 1] - matrix[1, 2]) / s
            y = (matrix[0, 1] + matrix[1, 0]) / s
            z = (matrix[0, 2] + matrix[2, 0]) / s
        elif index == 1:
            s = math.sqrt(
                1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            y, w = 0.25 * s, (matrix[0, 2] - matrix[2, 0]) / s
            x = (matrix[0, 1] + matrix[1, 0]) / s
            z = (matrix[1, 2] + matrix[2, 1]) / s
        else:
            s = math.sqrt(
                1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            z, w = 0.25 * s, (matrix[1, 0] - matrix[0, 1]) / s
            x = (matrix[0, 2] + matrix[2, 0]) / s
            y = (matrix[1, 2] + matrix[2, 1]) / s
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    return (x / norm, y / norm, z / norm, w / norm)


def estimate_cube_depth_pose(
    depth_image,
    box_xyxy,
    camera_matrix,
    *,
    side_length_m=0.057,
    roi_fraction=0.25,
    up_hint_xyz=None,
    floor_gap_fraction=0.05,
    floor_depth_fraction=1.2,
    floor_width_fraction=0.6,
    face_residual_m=0.004,
    minimum_up_share=0.2,
    minimum_up_samples=30,
    maximum_up_tilt_deg=35.0,
    minimum_face_share=0.25,
    minimum_face_samples=40,
    maximum_face_vertical_cosine=0.75,
    minimum_extent_fraction=0.7,
    maximum_extent_fraction=1.6,
    maximum_face_residual_m=0.005,
):
    """Return the camera-frame cube centre and frame, or raise if unsupported.

    ``depth_image`` is in metres, aligned to the image the box was measured in.
    ``up_hint_xyz`` is the up axis in the camera frame when the caller knows it
    (the robot's own frame does); the floor below the box is only fitted when
    that is absent.
    """
    depth = np.asarray(depth_image, dtype=np.float64)
    if depth.ndim != 2:
        raise CubeDepthPoseError('depth image must be two-dimensional')
    side = float(side_length_m)
    if side <= 0.0:
        raise CubeDepthPoseError('side_length_m must be positive')
    k = _camera_matrix(camera_matrix)
    x0, y0, x1, y1 = (float(value) for value in box_xyxy)
    width = x1 - x0

    if up_hint_xyz is None:
        # The floor below the box fixes "up": the box is tight around the cube, so
        # the region under it is not cube, and it is flat wherever it is visible.
        floor_v0 = y1 + float(floor_gap_fraction) * width
        floor_v1 = floor_v0 + float(floor_depth_fraction) * width
        _, floor_normals = _window_points(
            depth, k,
            x0 - float(floor_width_fraction) * width,
            x1 + float(floor_width_fraction) * width,
            floor_v0, floor_v1, minimum_samples=50)
        up, up_count, up_sample = _dominant_direction(floor_normals)
        if up @ _OPTICAL_DOWN > 0.0:
            up = -up
        tilt_deg = math.degrees(math.acos(
            float(np.clip(abs(up @ _OPTICAL_DOWN), -1.0, 1.0))))
        if tilt_deg > float(maximum_up_tilt_deg):
            raise CubeDepthPoseError(
                'the surface below the box is not a floor (%.1f deg off level)'
                % tilt_deg)
        # The region below the box is rarely all floor: the robot's own body and
        # the furniture around it are in the same band and have holes where the
        # sensor could not measure. The gate is therefore on the size of the flat
        # patch, not on how little else is there.
        up_share = float(up_count) / float(up_sample)
        if (up_count < int(minimum_up_samples)
                or up_share < float(minimum_up_share)):
            raise CubeDepthPoseError(
                'no flat floor below the box (%d of %d sampled normals agree)'
                % (up_count, up_sample))
        up_source = 'depth_floor'
    else:
        up = _unit(up_hint_xyz)
        if up @ _OPTICAL_DOWN > 0.0:
            up = -up
        up_share = None
        up_source = 'up_hint'
    tilt_deg = math.degrees(math.acos(
        float(np.clip(abs(up @ _OPTICAL_DOWN), -1.0, 1.0))))

    # The box's own vertical faces: the dominant direction that is not the floor
    # and not parallel to it.
    inner, whole = _box_windows(depth, box_xyxy, roi_fraction)
    _, box_normals = _window_points(
        depth, k, inner[0], inner[1], inner[2], inner[3], minimum_samples=200)
    usable = box_normals[np.isfinite(box_normals).all(axis=1)]
    vertical = usable[np.abs(usable @ up) < float(maximum_face_vertical_cosine)]
    if vertical.shape[0] < 100:
        raise CubeDepthPoseError('too few vertical-face pixels in the box')
    axis, face_count, face_sample = _dominant_direction(vertical)
    face_share = float(face_count) / float(face_sample)
    if (face_count < int(minimum_face_samples)
            or face_share < float(minimum_face_share)):
        raise CubeDepthPoseError(
            'no single vertical face dominates the box (%d of %d sampled '
            'normals agree)' % (face_count, face_sample))
    axis = _unit(axis - up * float(axis @ up))

    # The face is where the points are densest along the axis, not furthest: the
    # floor in front of the cube reaches further along a tilted axis.
    cube_points, _ = _window_points(
        depth, k, whole[0], whole[1], whole[2], whole[3], minimum_samples=200)
    if axis @ np.median(cube_points, axis=0) > 0.0:
        axis = -axis
    axis_b = _unit(np.cross(up, axis))

    along_axis = cube_points @ axis
    face_offset = _dense_band(along_axis)
    face_inliers = np.abs(along_axis - face_offset) <= float(face_residual_m)
    support = int(face_inliers.sum())
    face_points = cube_points[face_inliers]
    distance = float(np.median(face_points[:, 2]))
    expected_side_px = k[0, 0] * side / max(distance, 1e-6)
    if support < int(0.25 * expected_side_px * expected_side_px):
        raise CubeDepthPoseError(
            'the vertical face has too little support (%d px, expected about '
            '%d)' % (support, int(expected_side_px * expected_side_px)))
    face_offset = float(np.median(along_axis[face_inliers]))
    face_residual = float(
        np.median(np.abs(along_axis[face_inliers] - face_offset)))
    if face_residual > float(maximum_face_residual_m):
        raise CubeDepthPoseError(
            'the vertical face is not flat (mad %.4f m)' % face_residual)

    # The measured surface is the face itself, so the centre lies half a side
    # behind it; the face's in-plane extents say how much of it was visible.
    along_y = face_points @ axis_b
    along_z = face_points @ up
    extent_y = float(along_y.max() - along_y.min())
    extent_z = float(along_z.max() - along_z.min())
    for extent, name in ((extent_y, 'horizontal'), (extent_z, 'vertical')):
        if extent < float(minimum_extent_fraction) * side:
            raise CubeDepthPoseError(
                'the visible face is too small in the %s direction (%.3f m)'
                % (name, extent))
        if extent > float(maximum_extent_fraction) * side:
            raise CubeDepthPoseError(
                'the visible surface spans too much in the %s direction '
                '(%.3f m)' % (name, extent))

    centre = (
        axis * (face_offset - 0.5 * side)
        + axis_b * float(0.5 * (along_y.max() + along_y.min()))
        + up * float(0.5 * (along_z.max() + along_z.min()))
    )
    return CubeDepthPose(
        position_m=tuple(float(value) for value in centre),
        orientation_xyzw=_quaternion_from_axes(axis, axis_b, up),
        up_xyz=tuple(float(value) for value in up),
        up_source=up_source,
        up_share=up_share,
        up_tilt_deg=tilt_deg,
        face_share=face_share,
        face_support=support,
        face_residual_m=face_residual,
        in_plane_extent_m=extent_y,
        side_length_m=side,
    )
