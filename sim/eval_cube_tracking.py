#!/usr/bin/env python3
"""Measure the silhouette-refinement tracker against truth, from a good and a bad prior.

Recognition only. Two cases per distance: a prior at the truth, and one offset by about
a centimetre and five degrees, which is what a pose from the single-view solve looks
like in practice. The point is whether refinement converges from the worse one.

Run inside the lime container with the stack up:
    PYTHONPATH=/root/repo/yolo_ws/src/barcode_detector python3 sim/eval_cube_tracking.py
"""
from __future__ import annotations

import json
import math
import os
import sys

sys.path.insert(0, '/root/repo/yolo_ws/src')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from barcode_detector.cube_pose_tracking import (  # noqa: E402
    CubePoseTrackingError, refine_pose_from_silhouette)

import eval_common  # noqa: E402
import pose_math  # noqa: E402

MAX_CORNER_DISTANCE_PX = 25.0


def cases_from(truth_position, truth_orientation):
    """The two priors: at the truth, and a realistic step away from it."""
    return {
        'exact_prior': (list(truth_position), truth_orientation),
        'off_prior': ([truth_position[0] + 0.010, truth_position[1] + 0.005,
                       truth_position[2] + 0.015],
                      pose_math.multiply(
                          pose_math.axis_angle((0.0, 1.0, 0.0), 5.0),
                          truth_orientation)),
    }


def measure(distance, node):
    frame = node.wait_image()
    if frame is None:
        print('d=%.2f  no frame' % distance)
        return {'distance': distance, 'status': 'no frame'}
    image, frame_id, key = frame
    camera = node.camera_transform(frame_id)
    if camera is None:
        print('d=%.2f  no transform %s' % (distance, frame_id))
        return {'distance': distance, 'status': 'no transform'}

    truth_position = eval_common.cube_in_frame(distance, camera)
    truth_orientation = pose_math.quaternion_from_matrix(
        pose_math.matrix_from_quaternion(camera[1]))

    results = {}
    for name, (prior_position, prior_orientation) in cases_from(
            truth_position, truth_orientation).items():
        prior_error = math.dist(prior_position, truth_position)
        prior_angle = pose_math.angle_between(prior_orientation, truth_orientation)
        try:
            pose, evidence = refine_pose_from_silhouette(
                image, key, prior_position, prior_orientation,
                side_length_m=eval_common.CUBE_SIDE_M,
                maximum_corner_distance_px=MAX_CORNER_DISTANCE_PX)
        except CubePoseTrackingError as error:
            results[name] = {'prior_m': round(prior_error, 4),
                             'prior_deg': round(prior_angle, 2),
                             'status': 'refused: %s' % error}
            continue
        results[name] = {
            'prior_m': round(prior_error, 4),
            'prior_deg': round(prior_angle, 2),
            'refined_m': round(math.dist(pose.translation_m, truth_position), 4),
            'refined_deg': round(pose_math.angle_between(
                pose.orientation_xyzw, truth_orientation), 2),
            'matched': evidence['matched_vertices'],
        }
    print('d=%.2f  %s' % (distance, json.dumps(results, sort_keys=True)))
    return {'distance': distance, 'cases': results}


if __name__ == '__main__':
    eval_common.run_over_distances(measure)
