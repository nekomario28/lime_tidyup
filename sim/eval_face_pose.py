#!/usr/bin/env python3
"""Measure the single-view face pose on real sim frames.

Recognition only. For each distance it spawns the cube, grabs a camera frame, asks
the perception box for the visible face, and solves that face with IPPE. The colour
label decides which of the six faces it is, which is what fixes yaw: without it the
same square fits six cube poses. The table reports position error, the recovered
face, and whether the colour agreed with it.

Run inside the lime container with the stack up:
    PYTHONPATH=/root/repo/yolo_ws/src/barcode_detector python3 sim/eval_face_pose.py
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, '/root/repo/yolo_ws/src')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from barcode_detector.cube_single_view_pose import (  # noqa: E402
    FacePoseError, estimate_face_pose)

import eval_common  # noqa: E402

MAX_REPROJECTION_PX = 6.0


def measure(distance, node):
    """Solve the visible face and compare it with where the cube was spawned."""
    frame = node.wait_image()
    result = node.wait_result()
    if frame is None or result is None:
        print('d=%.2f  no frame or no detection' % distance)
        return {'distance': distance, 'status': 'no frame or no detection'}
    image, frame_id, key = frame
    camera = node.camera_transform(frame_id)
    if camera is None:
        print('d=%.2f  no transform' % distance)
        return {'distance': distance, 'status': 'no transform'}
    truth = eval_common.cube_in_frame(distance, camera)

    box = None
    best = 0.0
    for detection in result.get('detections', []):
        if detection.get('box_xyxy') and detection.get('confidence', 0.0) > best:
            best = detection['confidence']
            box = detection['box_xyxy']
    if box is None:
        print('d=%.2f  no box' % distance)
        return {'distance': distance, 'status': 'no box'}

    try:
        pose, evidence = estimate_face_pose(
            image, key, box, side_length_m=eval_common.CUBE_SIDE_M,
            maximum_reprojection_error_px=MAX_REPROJECTION_PX)
    except FacePoseError as error:
        print('d=%.2f  refused: %s' % (distance, error))
        return {'distance': distance, 'status': 'refused: %s' % error}

    error_m = math.dist(pose['position_m'], truth)
    print('d=%.2f  face %-3s label %-6s matched %-5s pos err %.4f m  reproj %.2f px'
          % (distance, pose['visible_face'], evidence['face_color_label'],
             evidence['color_matched'], error_m, pose['reprojection_error_px']))
    return {'distance': distance,
            'face': pose['visible_face'],
            'color_label': evidence['face_color_label'],
            'color_matched': evidence['color_matched'],
            'position_error_m': round(error_m, 4),
            'reprojection_px': round(pose['reprojection_error_px'], 2)}


if __name__ == '__main__':
    eval_common.run_over_distances(measure)
