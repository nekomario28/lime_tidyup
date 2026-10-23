#!/usr/bin/env python3
"""Measure cube recognition against ground truth in the sim.

Recognition only: it spawns the cube at a set of distances, reads the pose the
perception node publishes, and compares it with the cube's real pose. Nothing is
moved by the arm and nothing is grasped.

The comparison is done in base_link. The measured pose arrives in a camera
optical frame, so the published static transform is applied; the ground truth is
the cube's world pose from Gazebo, expressed in base_link with the robot's world
pose. The robot's model frame sits about base_link (z offset 0.06) and that
offset is subtracted, which is why the absolute numbers are quoted to the
centimetre while the *differences* between configurations are exact.

Every route in a message is measured and labelled by its own geometry id, so a
message carrying the depth centre as the primary pose and the face solve beside
it reports both errors rather than one of them twice.

Usage (inside the lime container, with the stack up):
    python3 sim/eval_cube_recognition.py [--distances 0.4 0.6 0.8 1.0]
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eval_common  # noqa: E402


def model_pose(name, attempts=3):
    """World pose of the robot, or None when this backend cannot be asked.

    MuJoCo has no `gz model`, but it does not need one: the scene gives the base a
    floating joint at the world origin, so the base is exactly there and the robot pose is
    known rather than guessed. Reporting it as the nominal 0.02 m settle offset (which is
    what the Gazebo fallback uses) would put 0.02 m of error into the truth -- measured,
    that turned a correct 0.5016 m reading into a 2.17 cm "error".
    """
    if os.environ.get('CCCSIM_BACKEND', 'gazebo') == 'mujoco':
        return {'x': 0.0, 'y': 0.0, 'z': 0.0}
    output = ''
    for _ in range(attempts):
        try:
            output = subprocess.run(['gz', 'model', '--model-name', name, '--info'],
                                    capture_output=True, text=True, timeout=25).stdout
        except Exception:  # noqa: BLE001
            continue
        if 'position {' in output:
            break
    values = {}
    lines = output.splitlines()
    for index, line in enumerate(lines):
        if line.strip() == 'position {':
            for offset in (1, 2, 3):
                if index + offset < len(lines) and ':' in lines[index + offset]:
                    key, _, value = lines[index + offset].partition(':')
                    try:
                        values[key.strip()] = float(value)
                    except ValueError:
                        return None
    return values or None


def measure(distance, node, robot):
    """Read the published routes and weigh each against where the cube was spawned."""
    result = node.wait_result()
    if result is None:
        print('d=%.2f  no result' % distance)
        return {'distance': distance, 'status': 'no result'}
    row = {'distance': distance,
           'detections': len(result.get('detections', []))}

    # The robot settles where physics puts it, which is not exactly the spawn pose, so
    # the truth uses the queried pose when Gazebo answered at all.
    truth = eval_common.cube_in_base_link(distance, robot)

    # Every route the message carries, keyed by its own geometry id: pose_3d is
    # whichever one the node published as primary, pose_3d_face the one beside it.
    # Selecting by field name instead reports one route's error under another's label.
    routes = {}
    for detection in result.get('detections', []):
        for candidate in (detection.get('pose_3d'), detection.get('pose_3d_face')):
            geometry = (candidate or {}).get('geometry_id')
            if geometry and len(candidate.get('position_m') or []) == 3:
                routes.setdefault(geometry,
                                  (candidate, detection.get('confidence')))

    route_errors = {}
    for geometry, (candidate, confidence) in routes.items():
        position = node.point_in_base_link(candidate['position_m'],
                                           candidate['frame_id'])
        if position is not None:
            route_errors[geometry] = (candidate, position,
                                      math.dist(position, truth))
    if not route_errors:
        row.update(status='no pose with a transform')
        print('d=%.2f  %d detections, no pose with a transform'
              % (distance, row['detections']))
        return row

    # The primary route is the one the node filled pose_3d with, which is the detection
    # whose primary pose this eval exists to weigh; the rest are companions.
    primary_geometry = next(
        ((geometry, detection.get('confidence'))
         for detection in result.get('detections', [])
         for geometry in [(detection.get('pose_3d') or {}).get('geometry_id')]
         if geometry in route_errors), None)
    if primary_geometry is None:
        primary_geometry = (sorted(route_errors)[0], None)
    geometry, confidence = primary_geometry
    candidate, position, error = route_errors[geometry]

    face = next(((geometry, values) for geometry, values in route_errors.items()
                 if 'face' in geometry), None)
    if face is not None:
        row.update(face_geometry=face[0],
                   face=[round(v, 4) for v in face[1][1]],
                   face_error_m=round(face[1][2], 4),
                   face_visible=face[1][0].get('visible_face'),
                   face_color=face[1][0].get('face_color_label'))
    row.update(status='ok',
               confidence=confidence,
               geometry=geometry,
               measured=[round(v, 4) for v in position],
               truth=[round(v, 4) for v in truth],
               error_m=round(error, 4),
               route_error_m={geometry: round(values[2], 4)
                              for geometry, values in route_errors.items()})
    print('d=%.2f  %s   face %-3s/%s'
          % (distance,
             '  '.join('%s err %.4f m' % (geometry, values[2])
                       for geometry, values in sorted(route_errors.items())),
             (row.get('face_visible') or '-'),
             (row.get('face_color') or '-')))
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--distances', nargs='*', type=float,
                        default=[0.4, 0.6, 0.8, 1.0])
    arguments = parser.parse_args()

    robot = model_pose('turtlebot3_lime_system')
    if robot is None:
        print('robot pose unavailable; assuming base_link at x=%.2f'
              % eval_common.ROBOT_X_M)
    else:
        print('robot base at x=%.4f y=%.4f z=%.4f'
              % (robot.get('x', 0.0), robot.get('y', 0.0), robot.get('z', 0.0)))
    nominal = (eval_common.ROBOT_X_M, 0.0, eval_common.BASE_Z_M)
    robot_base = (robot['x'], robot['y'],
                  robot['z'] + eval_common.BASE_Z_M) if robot else nominal
    eval_common.run_over_distances(
        lambda distance, node: measure(distance, node, robot_base),
        distances=arguments.distances)


if __name__ == '__main__':
    main()
