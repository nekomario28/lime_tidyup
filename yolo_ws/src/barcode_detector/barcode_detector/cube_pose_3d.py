"""
Estimate a cube camera-frame pose from eight ordered YOLO keypoints.

The geometry and acceptance gates are adapted from
``nekomario28/lime-interactive-transport`` Draft PR #4 (MIT).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol, Sequence


CANONICAL_VERTEX_LABELS = (
    'x_neg_y_neg_z_neg',
    'x_pos_y_neg_z_neg',
    'x_pos_y_pos_z_neg',
    'x_neg_y_pos_z_neg',
    'x_neg_y_neg_z_pos',
    'x_pos_y_neg_z_pos',
    'x_pos_y_pos_z_pos',
    'x_neg_y_pos_z_pos',
)

GEOMETRY_ID = 'cube-axis-v1'


class CubePose3DError(ValueError):
    """Raised when the available evidence cannot produce an accepted pose."""


@dataclass(frozen=True)
class PnpBackendResult:
    """Raw result returned by a PnP backend."""

    success: bool
    rotation_vector: tuple[float, float, float]
    translation_m: tuple[float, float, float]
    inlier_indices: tuple[int, ...]
    projected_points: tuple[tuple[float, float], ...]
    rotation_matrix: tuple[tuple[float, float, float], ...]


class PnpBackend(Protocol):
    """Backend interface used to keep geometry validation testable."""

    def solve(
        self,
        object_points: Sequence[tuple[float, float, float]],
        image_points: Sequence[tuple[float, float]],
        camera_matrix: Sequence[float],
        distortion: Sequence[float],
        maximum_reprojection_error_px: float,
    ) -> PnpBackendResult:
        """Solve the 3D-to-2D correspondence problem."""


@dataclass(frozen=True)
class CubePose3D:
    """Accepted camera-frame cube pose and its supporting quality evidence."""

    translation_m: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    reprojection_error_px: float
    correspondence_count: int
    inlier_count: int
    inlier_mask: tuple[int, ...]

    def as_dict(self, *, frame_id: str, timestamp: float, side_length_m: float):
        """Return a JSON-compatible representation for ``/cube_pose_result``."""
        return {
            'geometry_id': GEOMETRY_ID,
            'frame_id': frame_id,
            'timestamp': float(timestamp),
            'side_length_m': float(side_length_m),
            'position_m': list(self.translation_m),
            'orientation_xyzw': list(self.orientation_xyzw),
            'reprojection_error_px': self.reprojection_error_px,
            'correspondence_count': self.correspondence_count,
            'inlier_count': self.inlier_count,
            'inlier_mask': list(self.inlier_mask),
        }


def estimate_cube_pose_3d(
    keypoints_xy,
    keypoint_confidences,
    camera_matrix,
    distortion=(),
    *,
    side_length_m=0.057,
    minimum_keypoint_confidence=0.50,
    minimum_inliers=4,
    maximum_reprojection_error_px=4.0,
    backend=None,
):
    """Estimate and validate a camera-frame pose from ordered cube vertices."""
    side_length_m = float(side_length_m)
    minimum_keypoint_confidence = float(minimum_keypoint_confidence)
    maximum_reprojection_error_px = float(maximum_reprojection_error_px)

    if side_length_m <= 0.0:
        raise CubePose3DError('side_length_m must be positive')
    if len(keypoints_xy) != 8 or len(keypoint_confidences) != 8:
        raise CubePose3DError('exactly 8 ordered keypoints are required')
    if len(camera_matrix) != 9:
        raise CubePose3DError('camera matrix K must contain 9 values')

    calibration = tuple(float(value) for value in camera_matrix)
    distortion = tuple(float(value) for value in distortion)
    if not all(math.isfinite(value) for value in (*calibration, *distortion)):
        raise CubePose3DError('camera calibration contains non-finite values')
    if calibration[0] <= 0.0 or calibration[4] <= 0.0:
        raise CubePose3DError('camera focal lengths must be positive')

    vertices = _cube_vertices(side_length_m)
    selected = []
    for index, (vertex, point, confidence) in enumerate(
        zip(vertices, keypoints_xy, keypoint_confidences)
    ):
        if len(point) < 2:
            raise CubePose3DError('each keypoint must contain x and y')
        x = float(point[0])
        y = float(point[1])
        confidence = float(confidence)
        if not all(math.isfinite(value) for value in (x, y, confidence)):
            raise CubePose3DError('keypoint contains a non-finite value')
        if confidence >= minimum_keypoint_confidence and not (
            x == 0.0 and y == 0.0
        ):
            selected.append((index, vertex, (x, y)))

    if len(selected) < 4:
        raise CubePose3DError(
            'insufficient visible high-confidence correspondences'
        )

    if backend is None:
        backend = OpenCvPnpBackend()

    object_points = [vertex for _, vertex, _ in selected]
    image_points = [point for _, _, point in selected]
    result = backend.solve(
        object_points,
        image_points,
        calibration,
        distortion,
        maximum_reprojection_error_px,
    )

    if not result.success:
        raise CubePose3DError('solvePnPRansac failed')
    if len(result.projected_points) != len(image_points):
        raise CubePose3DError('backend returned an invalid projection count')
    if len(result.inlier_indices) < int(minimum_inliers):
        raise CubePose3DError('PnP inlier count is below the configured minimum')
    if not all(
        math.isfinite(value)
        for value in (*result.translation_m, *result.rotation_vector)
    ):
        raise CubePose3DError('PnP returned non-finite pose values')
    if result.translation_m[2] <= 0.0:
        raise CubePose3DError('object pose is behind the camera')

    squared_error = sum(
        (observed[0] - projected[0]) ** 2
        + (observed[1] - projected[1]) ** 2
        for observed, projected in zip(image_points, result.projected_points)
    )
    rms_error = math.sqrt(squared_error / len(image_points))
    if (
        not math.isfinite(rms_error)
        or rms_error > maximum_reprojection_error_px
    ):
        raise CubePose3DError(
            'PnP reprojection error exceeds the configured maximum'
        )

    inlier_mask = [0] * 8
    for selected_index in result.inlier_indices:
        if not 0 <= selected_index < len(selected):
            raise CubePose3DError(
                'backend returned an out-of-range inlier index'
            )
        inlier_mask[selected[selected_index][0]] = 1

    return CubePose3D(
        translation_m=tuple(float(value) for value in result.translation_m),
        orientation_xyzw=_quaternion_from_matrix(result.rotation_matrix),
        reprojection_error_px=rms_error,
        correspondence_count=len(selected),
        inlier_count=len(result.inlier_indices),
        inlier_mask=tuple(inlier_mask),
    )


def _cube_vertices(side_length_m):
    half = side_length_m / 2.0
    return (
        (-half, -half, -half),
        (+half, -half, -half),
        (+half, +half, -half),
        (-half, +half, -half),
        (-half, -half, +half),
        (+half, -half, +half),
        (+half, +half, +half),
        (-half, +half, +half),
    )


def _quaternion_from_matrix(matrix):
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        raise CubePose3DError('rotation matrix must be 3x3')
    m00, m01, m02 = matrix[0]
    m10, m11, m12 = matrix[1]
    m20, m21, m22 = matrix[2]
    trace = m00 + m11 + m22

    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (m21 - m12) / scale
        y = (m02 - m20) / scale
        z = (m10 - m01) / scale
    elif m00 > m11 and m00 > m22:
        scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / scale
        x = 0.25 * scale
        y = (m01 + m10) / scale
        z = (m02 + m20) / scale
    elif m11 > m22:
        scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / scale
        x = (m01 + m10) / scale
        y = 0.25 * scale
        z = (m12 + m21) / scale
    else:
        scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / scale
        x = (m02 + m20) / scale
        y = (m12 + m21) / scale
        z = 0.25 * scale

    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0 or not math.isfinite(norm):
        raise CubePose3DError('rotation matrix produced an invalid quaternion')
    return (x / norm, y / norm, z / norm, w / norm)


class OpenCvPnpBackend:
    """OpenCV implementation loaded lazily for the ROS runtime."""

    def solve(
        self,
        object_points,
        image_points,
        camera_matrix,
        distortion,
        maximum_reprojection_error_px,
    ):
        try:
            import cv2
            import numpy as np
        except ImportError as error:
            raise CubePose3DError(
                'OpenCV and NumPy are required for runtime PnP'
            ) from error

        object_array = np.asarray(object_points, dtype=np.float64)
        image_array = np.asarray(image_points, dtype=np.float64)
        camera_array = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
        distortion_array = np.asarray(distortion, dtype=np.float64)

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_array,
            image_array,
            camera_array,
            distortion_array,
            iterationsCount=100,
            reprojectionError=float(maximum_reprojection_error_px),
            confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success or inliers is None:
            return PnpBackendResult(
                success=False,
                rotation_vector=(0.0, 0.0, 0.0),
                translation_m=(0.0, 0.0, 0.0),
                inlier_indices=(),
                projected_points=(),
                rotation_matrix=(
                    (1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (0.0, 0.0, 1.0),
                ),
            )

        projected, _ = cv2.projectPoints(
            object_array,
            rvec,
            tvec,
            camera_array,
            distortion_array,
        )
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        return PnpBackendResult(
            success=True,
            rotation_vector=tuple(float(value) for value in rvec.reshape(-1)),
            translation_m=tuple(float(value) for value in tvec.reshape(-1)),
            inlier_indices=tuple(int(value) for value in inliers.reshape(-1)),
            projected_points=tuple(
                (float(point[0]), float(point[1]))
                for point in projected.reshape(-1, 2)
            ),
            rotation_matrix=tuple(
                tuple(float(value) for value in row)
                for row in rotation_matrix.tolist()
            ),
        )
