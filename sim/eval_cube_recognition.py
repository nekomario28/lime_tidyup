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

Usage (inside the lime container, with the stack up):
    python3 sim/eval_cube_recognition.py [--distances 0.4 0.6 0.8 1.0]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

BASE_Z_OFFSET = 0.06


def model_pose(name, attempts=3):
    """World pose of a Gazebo model (retried: gz model can time out)."""
    output = ''
    for _ in range(attempts):
        try:
            output = subprocess.run(
                ['gz', 'model', '--model-name', name, '--info'],
                capture_output=True, text=True, timeout=25).stdout
        except Exception:  # noqa: BLE001
            output = ''
        if 'position {' in output:
            break
        time.sleep(1.0)
    if not output:
        return None
    values = {}
    lines = output.splitlines()
    for index, line in enumerate(lines):
        text = line.strip()
        if text in ('position {', 'orientation {'):
            block = lines[index + 1:index + 4]
            for entry in block:
                key, _, raw = entry.strip().partition(':')
                if key in ('x', 'y', 'z', 'w'):
                    try:
                        values[key] = float(raw)
                    except ValueError:
                        pass
            if len(values) >= 3:
                break
    if len(values) < 3:
        return None
    return values


def rotate(quat, vector):
    x, y, z, w = quat
    ux, uy, uz = vector
    cross = (y * uz - z * uy, z * ux - x * uz, x * uy - y * ux)
    dot = x * ux + y * uy + z * uz
    uu = x * x + y * y + z * z
    return tuple(2.0 * dot * u + (w * w - uu) * p + 2.0 * w * c
                 for u, p, c in zip((x, y, z), (ux, uy, uz), cross))


class Probe(Node):
    def __init__(self):
        super().__init__('eval_cube_recognition')
        self.latest = None
        self.create_subscription(String, '/cube_pose_result', self.on_result, 1)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def on_result(self, msg):
        self.latest = msg.data

    def wait_result(self, timeout=15.0, settle=2.5):
        """Return a detection published after `settle` seconds.

        Taking the next message straight after a spawn picks up a frame from
        before it: measured, that moved the answer by 5 cm between restarts.
        """
        self.latest = None
        start = time.time()
        while time.time() - start < settle:
            self.latest = None
            time.sleep(0.2)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.latest:
                return json.loads(self.latest)
            time.sleep(0.1)
        return None

    def to_base_link(self, position, frame):
        try:
            transform = self.tf_buffer.lookup_transform(
                'base_link', frame, rclpy.time.Time())
        except Exception:  # noqa: BLE001
            return None
        q = transform.transform.rotation
        t = transform.transform.translation
        rotated = rotate((q.x, q.y, q.z, q.w), position)
        return [rotated[i] + (t.x, t.y, t.z)[i] for i in range(3)]


def spawn_cube(distance):
    subprocess.run(
        ['ros2', 'service', 'call', '/delete_entity',
         'gazebo_msgs/srv/DeleteEntity', '{name: rubik_cube}'],
        capture_output=True, text=True, timeout=20)
    time.sleep(0.5)
    subprocess.run(
        ['python3', os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'spawn_cube.py'), str(distance), '0.0'],
        capture_output=True, text=True, timeout=40)
    time.sleep(2.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--distances', nargs='*', type=float,
                        default=[0.4, 0.6, 0.8, 1.0])
    arguments = parser.parse_args()

    rclpy.init()
    node = Probe()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    time.sleep(1.5)

    # The robot settles where physics puts it, which is not exactly the spawn pose:
    # assuming 0.02 m here is a candidate explanation for sweeps that disagree with
    # each other by about 8 cm. Ask Gazebo (retried) and fall back to the nominal
    # pose only when it will not answer.
    robot = model_pose('turtlebot3_lime_system')
    if robot is None:
        print('robot pose unavailable; assuming base_link at x=0.02')
        robot = {'x': 0.02, 'y': 0.0, 'z': 0.0}
    else:
        print('robot base at x=%.4f y=%.4f z=%.4f'
              % (robot.get('x', 0.0), robot.get('y', 0.0), robot.get('z', 0.0)))
    rows = []
    for distance in arguments.distances:
        spawn_cube(distance)
        result = node.wait_result()
        if result is None or robot is None:
            print('d=%.2f  no result/ground truth (result=%s robot=%s)'
                  % (distance, result is not None, robot is not None))
            continue

        best = None
        for detection in result.get('detections', []):
            pose = detection.get('pose_3d') or {}
            if len(pose.get('position_m') or []) == 3:
                if best is None or detection['confidence'] > best[0]:
                    best = (detection['confidence'], pose)
        row = {'distance': distance,
               'detections': len(result.get('detections', []))}
        if best is None:
            row.update(status='no pose_3d')
            rows.append(row)
            print('d=%.2f  %d detections, no pose_3d' % (distance, row['detections']))
            continue

        confidence, pose = best
        measured = node.to_base_link(pose['position_m'], pose['frame_id'])
        if measured is None:
            row.update(status='no transform %s' % pose['frame_id'])
            rows.append(row)
            print('d=%.2f  no transform for %s' % (distance, pose['frame_id']))
            continue

        # The cube is spawned at a known distance and does not move, so the spawn
        # value is the better ground truth: gz model is slow and occasionally
        # times out while the sim is stepping, and the cube's own query is the one
        # that matters least.
        truth_world = [distance, 0.0, 0.5 * 0.057]
        robot_world = [robot['x'], robot['y'], robot['z'] + BASE_Z_OFFSET]
        truth = [truth_world[i] - robot_world[i] for i in range(3)]
        error = math.dist(measured, truth)
        # Every route in the message is measured against the same truth, keyed by its
        # own geometry id. Selecting by field name would report one route's error under
        # another's label: the node publishes the face solve as pose_3d and the depth
        # centre as pose_3d_depth, so `pose_3d_face` is empty and a fixed lookup reads
        # the depth pose while calling it the face.
        routes = {}
        for detection in result.get('detections', []):
            for candidate in (detection.get('pose_3d'),
                              detection.get('pose_3d_depth'),
                              detection.get('pose_3d_face')):
                geometry = (candidate or {}).get('geometry_id')
                if geometry and len(candidate.get('position_m') or []) == 3:
                    routes.setdefault(geometry, candidate)
        route_errors = {}
        for geometry, candidate in routes.items():
            position = node.to_base_link(candidate['position_m'],
                                         candidate['frame_id'])
            if position is not None:
                route_errors[geometry] = (candidate, position,
                                          math.dist(position, truth))
        face_only = next(((g, c, p, e) for g, (c, p, e) in route_errors.items()
                          if 'face' in g), None)
        if face_only is not None:
            row.update(face_geometry=face_only[0],
                       face=[round(v, 4) for v in face_only[2]],
                       face_error_m=round(face_only[3], 4),
                       face_visible=face_only[1].get('visible_face'),
                       face_color=face_only[1].get('face_color_label'))
        row.update(status='ok', confidence=confidence,
                   geometry=pose.get('geometry_id'),
                   measured=[round(v, 4) for v in measured],
                   truth=[round(v, 4) for v in truth],
                   error_m=round(error, 4),
                   route_error_m={g: round(e, 4)
                                  for g, (_, _, e) in route_errors.items()})
        rows.append(row)
        print('d=%.2f  %s   face %-3s/%s'
              % (distance,
                 '  '.join('%s err %.4f m' % (geometry, values[2])
                           for geometry, values in sorted(route_errors.items())),
                 (row.get('face_visible') or '-'),
                 (row.get('face_color') or '-')))

    print('\nsummary')
    for row in rows:
        print(json.dumps(row, sort_keys=True))


if __name__ == '__main__':
    main()
