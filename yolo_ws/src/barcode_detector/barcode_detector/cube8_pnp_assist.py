"""Seven-observation PnP helper and hidden-vertex reprojection for Cube8."""
from __future__ import annotations
import math
from typing import Mapping, Sequence
from .cube8_geometry import CANONICAL_VERTEX_LABELS, PoseKeypoint2D, cube_vertices
from .cube8_pnp import CameraModel, Cube8PnpError, CubePose3D, estimate_cube_pose


def estimate_pose_from_observed_vertices(
    observations: Mapping[int, Sequence[float]],
    camera: CameraModel,
    *,
    side_length_m: float = 0.057,
    maximum_reprojection_error_px: float = 4.0,
) -> CubePose3D:
    """Solve PnP from only supplied canonical observations; missing hidden point stays excluded."""
    invalid = [
        i for i in observations if isinstance(
            i, bool) or not isinstance(
            i, int) or not 0 <= i < 8]
    if invalid:
        raise Cube8PnpError(
            f"canonical vertex indices outside [0,7]: {invalid}")
    keypoints = []
    for index, label in enumerate(CANONICAL_VERTEX_LABELS):
        if index not in observations:
            keypoints.append(PoseKeypoint2D(label, 0.0, 0.0, 0.0, 0))
            continue
        raw = observations[index]
        if len(raw) != 2:
            raise Cube8PnpError(f"observation {index} must contain x,y")
        x, y = float(raw[0]), float(raw[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise Cube8PnpError(f"observation {index} is non-finite")
        keypoints.append(PoseKeypoint2D(label, x, y, 1.0, 2))
    return estimate_cube_pose(
        tuple(keypoints), camera,
        side_length_m=side_length_m,
        minimum_keypoint_confidence=0.5,
        minimum_correspondences=4,
        minimum_inliers=4,
        maximum_reprojection_error_px=maximum_reprojection_error_px,
    )


def project_vertex(pose: CubePose3D, vertex_index: int,
                   camera: CameraModel) -> tuple[float, float]:
    """Project a canonical cube vertex through an already estimated pose."""
    import cv2
    import numpy as np
    camera.validate()
    if isinstance(
            vertex_index,
            bool) or not isinstance(
            vertex_index,
            int) or not 0 <= vertex_index < 8:
        raise Cube8PnpError("vertex_index must be an integer in [0,7]")
    vertex = cube_vertices(pose.side_length_m)[vertex_index]
    projected, _ = cv2.projectPoints(
        np.asarray([(vertex.x, vertex.y, vertex.z)], dtype=np.float64),
        np.asarray(pose.rotation_vector, dtype=np.float64).reshape(3, 1),
        np.asarray(pose.translation_m, dtype=np.float64).reshape(3, 1),
        np.asarray(camera.k, dtype=np.float64).reshape(3, 3),
        np.asarray(camera.d, dtype=np.float64),
    )
    point = projected.reshape(-1, 2)[0]
    result = (float(point[0]), float(point[1]))
    if not all(math.isfinite(v) for v in result):
        raise Cube8PnpError("projected vertex is non-finite")
    return result
