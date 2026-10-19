"""depth_blob -> model-free pose, on a Gazebo depth frame.

Evidence for doc/design/2026-10-20-yolo-free-realtime-perception.md. Add a depth
sensor to the cube8 sim camera, spawn the 57 mm cube at 0.35 m, then run the
production path with no detector:

    estimate_model_free_pose(image, K, depth_image=depth,
                             bbox_source='depth_blob')

Measured: pose accepted, translation (-0.0004, -0.0001, 0.351) m against a
(0, 0, 0.35) m truth = 1.05 mm, RMS 1.03 px, colour resolved
(white/blue/orange). The truth is converted into the camera optical frame, which
is the frame the pose is expressed in.
"""
import json
import math
import sys

import cv2
import numpy as np

sys.path.insert(0, '/dev2/yolo_ws/src/barcode_detector')
from barcode_detector.cube_model_free_pose import (  # noqa: E402
    ModelFreePoseError,
    estimate_model_free_pose,
)

EV = '/work/.sim/evidence/depth'
K = (1060.9338813153618, 0.0, 640.5,
     0.0, 1060.9338813153618, 360.5,
     0.0, 0.0, 1.0)


def main():
    image = cv2.imread(f'{EV}/frame.png')
    depth = np.load(f'{EV}/depth.npy')
    truth = json.load(open(f'{EV}/truth.json'))
    def quat_matrix(q):
        x, y, z, w = q
        n = math.sqrt(x * x + y * y + z * z + w * w)
        x, y, z, w = x / n, y / n, z / n, w / n
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])

    # the pose is expressed in the camera optical frame, so the truth must be too
    rotation_camera = quat_matrix(truth['world_from_camera']['qxyzw'])
    delta = (np.array(truth['world_from_cube']['xyz'])
             - np.array(truth['world_from_camera']['xyz']))
    trend = tuple(float(v) for v in rotation_camera.T @ delta)
    row = {
        'depth_finite': int(np.isfinite(depth).sum()),
        'truth_camera_from_cube_m': [round(v, 4) for v in trend],
    }
    try:
        pose = estimate_model_free_pose(
            image, K, depth_image=depth, bbox_source='depth_blob')
    except ModelFreePoseError as error:
        row['pose'] = {'status': 'rejected', 'reason': str(error)}
    else:
        error_mm = math.sqrt(sum(
            (got - want) ** 2
            for got, want in zip(pose.translation_m, trend)
        )) * 1000.0
        row['pose'] = {
            'status': 'accepted',
            'translation_m': [round(v, 4) for v in pose.translation_m],
            'translation_error_mm': round(error_mm, 2),
            'rms_px': round(pose.reprojection_error_px, 3),
            'bbox_source': pose.bbox_source,
            'bbox_quality': round(pose.bbox_quality, 3),
            'bbox': [round(v, 1) for v in pose.bbox_xyxy],
            'resolved': pose.color_fixed_semantics_resolved,
            'colors': pose.visible_face_colors,
            'label_swaps': list(pose.label_swaps),
        }
    print(json.dumps(row, indent=2))


if __name__ == '__main__':
    main()
