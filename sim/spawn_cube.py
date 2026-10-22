#!/usr/bin/env python3
"""Spawn the 57 mm Rubik cube as a dynamic body in front of the robot."""
import math
import os
import sys

import rclpy
from gazebo_msgs.srv import SpawnEntity
from geometry_msgs.msg import Pose as RosPose
from rclpy.node import Node

# Generate the cube from the repository generator instead of shipping a copy, so
# the sim target cannot drift from the asset the perception code is tuned on.
# The module is loaded by path: an installed barcode_detector earlier on
# PYTHONPATH would otherwise win and may predate the generator.
RUBIK_GENERATOR_CANDIDATES = (
    '/root/yolo_ws/src/barcode_detector/barcode_detector/rubik_pose_cube.py',
    '/root/repo/yolo_ws/src/barcode_detector/barcode_detector/rubik_pose_cube.py',
)


def load_cube_generator():
    import importlib.util

    for candidate in RUBIK_GENERATOR_CANDIDATES:
        if os.path.isfile(candidate):
            spec = importlib.util.spec_from_file_location('rubik_pose_cube', candidate)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise SystemExit('rubik_pose_cube.py not found; is yolo_ws mounted?')


def main():
    distance = float(sys.argv[1]) if len(sys.argv) > 1 else 0.34
    lateral = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    # Yaw matters for the grasp: the jaws straddle two opposite faces, so a cube turned
    # 45 degrees presents its diagonal at them. Default 0 keeps the old behaviour.
    yaw_deg = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    xml = load_cube_generator().build_model_sdf(static=False)
    rclpy.init()
    node = Node('spawn_cube')
    client = node.create_client(SpawnEntity, '/spawn_entity')
    if not client.wait_for_service(timeout_sec=30.0):
        raise SystemExit('/spawn_entity unavailable')
    request = SpawnEntity.Request()
    request.name = 'rubik_cube'
    request.xml = xml
    request.robot_namespace = ''
    request.initial_pose = RosPose()
    request.initial_pose.position.x = distance
    request.initial_pose.position.y = lateral
    request.initial_pose.position.z = 0.057 / 2.0 + 0.001
    half = math.radians(yaw_deg) / 2.0
    request.initial_pose.orientation.z = math.sin(half)
    request.initial_pose.orientation.w = math.cos(half)
    request.reference_frame = 'world'
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=30.0)
    result = future.result()
    if result is None or not result.success:
        status = 'no response' if result is None else result.status_message
        raise SystemExit(f'spawn failed: {status}')
    print(f'cube spawned at {distance}, {lateral}, yaw {yaw_deg:.0f} deg')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
