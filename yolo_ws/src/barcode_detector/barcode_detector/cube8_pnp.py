"""Eight-point cube PnP. Computation only; does not command a robot."""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Sequence
from .cube8_geometry import CANONICAL_VERTEX_LABELS, GEOMETRY_ID, PoseKeypoint2D, cube_vertices


class Cube8PnpError(ValueError):
    pass


@dataclass(frozen=True)
class CameraModel:
    k: tuple[float, ...]
    d: tuple[float, ...] = ()

    def validate(self):
        if len(self.k) != 9:
            raise Cube8PnpError("camera K must contain 9 values")
        if not all(math.isfinite(float(v)) for v in (*self.k, *self.d)):
            raise Cube8PnpError("non-finite camera calibration")
        if self.k[0] <= 0 or self.k[4] <= 0:
            raise Cube8PnpError("camera focal lengths must be positive")


@dataclass(frozen=True)
class CubePose3D:
    geometry_id: str
    side_length_m: float
    translation_m: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    rotation_vector: tuple[float, float, float]
    reprojection_error_px: float
    correspondence_count: int
    inlier_count: int
    inlier_mask: tuple[int, ...]

    def as_dict(self):
        return {
            "geometry_id": self.geometry_id,
            "side_length_m": self.side_length_m,
            "translation_m": list(self.translation_m),
            "orientation_xyzw": list(self.orientation_xyzw),
            "rotation_vector": list(self.rotation_vector),
            "reprojection_error_px": self.reprojection_error_px,
            "correspondence_count": self.correspondence_count,
            "inlier_count": self.inlier_count,
            "inlier_mask": list(self.inlier_mask),
        }


def estimate_cube_pose(
        keypoints: Sequence[PoseKeypoint2D],
        camera: CameraModel,
        *,
        side_length_m: float = 0.057,
        minimum_keypoint_confidence: float = 0.50,
        minimum_correspondences: int = 4,
        minimum_inliers: int = 4,
        maximum_reprojection_error_px: float = 4.0) -> CubePose3D:
    import cv2
    import numpy as np
    camera.validate()
    if len(keypoints) != 8 or tuple(
            p.label for p in keypoints) != CANONICAL_VERTEX_LABELS:
        raise Cube8PnpError(
            "exactly 8 canonical ordered keypoints are required")
    vertices = cube_vertices(float(side_length_m))
    selected = [(i, v, p) for i, (v, p) in enumerate(zip(vertices, keypoints))
                if p.visibility == 2 and p.confidence >= float(minimum_keypoint_confidence)]
    if len(selected) < int(minimum_correspondences):
        raise Cube8PnpError(
            "insufficient visible high-confidence correspondences")
    obj = np.asarray([(v.x, v.y, v.z)
                     for _, v, _ in selected], dtype=np.float64)
    img = np.asarray([(p.x, p.y) for _, _, p in selected], dtype=np.float64)
    K = np.asarray(camera.k, dtype=np.float64).reshape(3, 3)
    d = np.asarray(camera.d, dtype=np.float64)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj, img, K, d, iterationsCount=100,
        reprojectionError=float(maximum_reprojection_error_px),
        confidence=0.99, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or inliers is None:
        raise Cube8PnpError("solvePnPRansac failed")
    inlier_indices = tuple(int(x) for x in inliers.reshape(-1))
    if len(inlier_indices) < int(minimum_inliers):
        raise Cube8PnpError("PnP inlier count below minimum")
    projected, _ = cv2.projectPoints(obj, rvec, tvec, K, d)
    projected = projected.reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum((img - projected)**2, axis=1))))
    if not math.isfinite(rms) or rms > float(maximum_reprojection_error_px):
        raise Cube8PnpError("reprojection error exceeds maximum")
    R, _ = cv2.Rodrigues(rvec)
    quat = _quat_from_rotation(R)
    t = tuple(float(v) for v in tvec.reshape(-1))
    if t[2] <= 0:
        raise Cube8PnpError("pose is behind camera")
    mask = [0] * 8
    for idx in inlier_indices:
        mask[selected[idx][0]] = 1
    return CubePose3D(
        GEOMETRY_ID, float(side_length_m), t, quat,
        tuple(float(v) for v in rvec.reshape(-1)),
        rms, len(selected), len(inlier_indices), tuple(mask))


def _quat_from_rotation(R) -> tuple[float, float, float, float]:
    import numpy as np
    M = np.asarray(R, dtype=float)
    m00, m01, m02 = M[0]
    m10, m11, m12 = M[1]
    m20, m21, m22 = M[2]
    tr = m00 + m11 + m22
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = .25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1 + m00 - m11 - m22) * 2
        w = (m21 - m12) / s
        x = .25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1 + m11 - m00 - m22) * 2
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = .25 * s
        z = (m12 + m21) / s
    else:
        s = math.sqrt(1 + m22 - m00 - m11) * 2
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = .25 * s
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(n) or n < 1e-12:
        raise Cube8PnpError("invalid rotation quaternion")
    return (x / n, y / n, z / n, w / n)
