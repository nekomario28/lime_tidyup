"""Render the production overlay for a sim frame and save it for review."""
import json
import math
import sys

import cv2
import numpy as np

sys.path.insert(0, '/dev2/yolo_ws/src/barcode_detector')
from barcode_detector.cube_model_free_pose import (  # noqa: E402
    ModelFreePoseError, estimate_model_free_pose)
from barcode_detector.cube_overlay import draw_cube_overlay  # noqa: E402

K = (1060.9338813153618, 0.0, 640.5,
     0.0, 1060.9338813153618, 360.5,
     0.0, 0.0, 1.0)
SIDE = 0.057


def main():
    out_dir = sys.argv[1]
    cases = [
        ('sim-depth', '/work/.sim/evidence/depth/frame.png',
         '/work/.sim/evidence/depth/depth.npy'),
        ('sim-depth-flat', '/work/.sim/evidence/depth/frame.png', None),
    ]
    for name, image_path, depth_path in cases:
        image = cv2.imread(image_path)
        depth = np.load(depth_path) if depth_path else None
        try:
            pose = estimate_model_free_pose(
                image, K, depth_image=depth, bbox_source='depth_blob')
        except ModelFreePoseError as error:
            annotated = draw_cube_overlay(
                image, camera_matrix=K, status='rejected', reason=str(error),
                bbox_source='depth_blob' if depth is not None else 'none')
            print(name, 'rejected:', error)
        else:
            pose_dict = {
                'position_m': list(pose.translation_m),
                'orientation_xyzw': list(pose.orientation_xyzw),
            }
            face_regions = {
                slot: evidence.get('quad')
                for slot, evidence in (pose.face_evidence or {}).items()
                if isinstance(evidence, dict) and evidence.get('quad')
            }
            annotated = draw_cube_overlay(
                image, camera_matrix=K, pose=pose_dict,
                bbox_xyxy=pose.bbox_xyxy, bbox_source=pose.bbox_source,
                bbox_quality=pose.bbox_quality,
                face_regions=face_regions,
                observed_colors=pose.visible_face_colors,
                visible_faces=pose.visible_faces,
                status=('accepted' if pose.color_fixed_semantics_resolved
                        else 'unresolved'),
                reason=pose.unresolved_reason,
                reprojection_error_px=pose.reprojection_error_px,
                color_resolved=pose.color_fixed_semantics_resolved,
                label_swaps=pose.label_swaps, side_length_m=SIDE)
            print(name, 'accepted:', [round(v, 4) for v in pose.translation_m],
                  'rms', round(pose.reprojection_error_px, 2),
                  'resolved', pose.color_fixed_semantics_resolved)
        path = f'{out_dir}/{name}.jpg'
        cv2.imwrite(path, annotated)
        print('  wrote', path)


if __name__ == '__main__':
    main()
