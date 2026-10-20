#!/usr/bin/env python3
"""Attach the cube to link7 with the project's LinkAttacher service."""
import sys

import rclpy
from linkattacher_msgs.srv import AttachLink
from rclpy.node import Node


def main():
    robot = sys.argv[1] if len(sys.argv) > 1 else 'turtlebot3_lime_system'
    link = sys.argv[2] if len(sys.argv) > 2 else 'gripper_left_link'
    model = sys.argv[3] if len(sys.argv) > 3 else 'rubik_cube'
    target = sys.argv[4] if len(sys.argv) > 4 else 'body'
    rclpy.init()
    node = Node('attach_cube')
    client = node.create_client(AttachLink, '/ATTACHLINK')
    if not client.wait_for_service(timeout_sec=20.0):
        raise SystemExit('/ATTACHLINK unavailable')
    request = AttachLink.Request()
    request.model1_name = robot
    request.link1_name = link
    request.model2_name = model
    request.link2_name = target
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=20.0)
    result = future.result()
    print('attach:', result)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
