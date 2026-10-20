#!/usr/bin/env python3
"""Spawn the 57 mm Rubik cube as a dynamic body in front of the robot."""
import sys

import rclpy
from gazebo_msgs.srv import SpawnEntity
from geometry_msgs.msg import Pose as RosPose
from rclpy.node import Node

# Generate the cube from the repository generator instead of shipping a copy, so
# the sim target cannot drift from the asset the perception code is tuned on.
sys_path = '/root/repo/yolo_ws/src/barcode_detector'
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)


def main():
    distance = float(sys.argv[1]) if len(sys.argv) > 1 else 0.34
    lateral = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    from barcode_detector.rubik_pose_cube import build_model_sdf

    xml = build_model_sdf(static=False)
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
    request.initial_pose.orientation.w = 1.0
    request.reference_frame = 'world'
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=30.0)
    result = future.result()
    if result is None or not result.success:
        status = 'no response' if result is None else result.status_message
        raise SystemExit(f'spawn failed: {status}')
    print(f'cube spawned at {distance}, {lateral}')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
