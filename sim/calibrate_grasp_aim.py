#!/usr/bin/env python3
"""Measure the aim offset that puts the pads on the cube, per pitch and dz.

The grasp aims the pads with `grasp_pose(point, pitch, dz)`, where dz shifts the
target down. dz was -0.03 because it worked once; this measures what it should be at
the pre-grasp distance: for each dz it poses the arm, reads the pads from TF, and
reports the residual against the cube the perception reports.

Recognition only in the sense that nothing is closed or lifted -- the arm just takes
a pose and is measured.

Run inside the lime container with the stack up:
    python3 sim/calibrate_grasp_aim.py [distance_m]
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

sys.path.insert(0, '/root/repo/sim')

import grasp_cube  # noqa: E402

DISTANCES = (0.26,)


class Probe(Node):
    def __init__(self):
        super().__init__('calibrate_grasp_aim')
        self.latest = None
        self.create_subscription(String, '/cube_pose_result', self.on_result, 1)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def on_result(self, msg):
        self.latest = msg.data

    def cube_in_base_link(self, timeout=15.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.latest:
                data = json.loads(self.latest)
                best = None
                for detection in data.get('detections', []):
                    pose = (detection.get('pose_3d_face')
                            or detection.get('pose_3d') or {})
                    if len(pose.get('position_m') or []) == 3:
                        if best is None or detection['confidence'] > best[0]:
                            best = (detection['confidence'], pose)
                if best is not None:
                    return grasp_cube.Grasper.to_base_link(self, best[1])
            time.sleep(0.1)
        return None


def main():
    distances = [float(value) for value in sys.argv[1:]] or list(DISTANCES)
    rclpy.init()
    node = Probe()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    time.sleep(1.5)

    # the arm side of the grasp is the same code the run uses, so the measurement
    # describes the real path rather than a copy of it
    grasper = grasp_cube.Grasper()

    for distance in distances:
        subprocess.run(['ros2', 'service', 'call', '/delete_entity',
                        'gazebo_msgs/srv/DeleteEntity', '{name: rubik_cube}'],
                       capture_output=True, text=True, timeout=20)
        time.sleep(0.5)
        subprocess.run(['python3', '/root/repo/sim/spawn_cube.py', str(distance),
                        '0.0'], capture_output=True, text=True, timeout=40)
        time.sleep(2.5)

        cube = node.cube_in_base_link()
        if cube is None:
            print('no cube pose; is perception running?')
            continue
        print('distance %.2f  cube at %.3f %.3f %.3f'
              % (distance, cube[0], cube[1], cube[2]))
        grasper.send_gripper(0.019)
        time.sleep(0.8)

        for pitch in (150.0, 120.0):
            for dz in (-0.02, -0.03, -0.04, -0.05, -0.06):
                target = grasp_cube.grasp_pose(cube, pitch, dz)
                joints = grasper.solve_ik(target, grasp_cube.pitch_quat(pitch))
                if joints is None:
                    print('  pitch %3.0f dz %+.2f  no ik' % (pitch, dz))
                    continue
                grasper.send_arm(joints, seconds=4.0)
                time.sleep(0.8)
                pads = grasper.fingertip_midpoint()
                if pads is None:
                    print('  pitch %3.0f dz %+.2f  no fingertip tf' % (pitch, dz))
                    continue
                residual = [pads[i] - cube[i] for i in range(3)]
                print('  pitch %3.0f dz %+.2f  pads %.3f %.3f %.3f  '
                      'residual x %+.3f y %+.3f z %+.3f'
                      % (pitch, dz, pads[0], pads[1], pads[2],
                         residual[0], residual[1], residual[2]))


if __name__ == '__main__':
    main()
