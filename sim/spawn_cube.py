#!/usr/bin/env python3
"""Put the 57 mm Rubik cube on the floor in front of the robot.

Two backends, one contract: on Gazebo the cube does not exist until this asks the
`/spawn_entity` service to build it, and it is deleted first so a re-run replaces it. On
MuJoCo the cube is part of the scene and has a free joint, so the engine's
`set_free_joint_state` service -- documented as "similar to Gazebo's set_entity_state" --
moves that body instead. The caller (the eval tools, or bin/cccsim's cube stage) does not
have to know which.
"""
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


MUJOCO_SERVICE = '/mujoco_ros2_control_node/set_free_joint_state'
CUBE_BODY = 'cube'


def place_with_mujoco(x, y, z, yaw_deg):
    """Move the scene's cube body to this pose through the MuJoCo control node."""
    import math

    from geometry_msgs.msg import PoseStamped
    from mujoco_ros2_control_msgs.msg import FreeJointState
    from mujoco_ros2_control_msgs.srv import SetFreeJointState

    node = Node('spawn_cube_mujoco')
    client = node.create_client(SetFreeJointState, MUJOCO_SERVICE)
    if not client.wait_for_service(timeout_sec=20.0):
        raise SystemExit('%s unavailable (is the MuJoCo control node running?)'
                         % MUJOCO_SERVICE)
    half = math.radians(yaw_deg) / 2.0
    entry = FreeJointState()
    entry.name = CUBE_BODY
    entry.pose = PoseStamped()
    entry.pose.header.frame_id = 'world'
    entry.pose.pose.position.x = x
    entry.pose.pose.position.y = y
    entry.pose.pose.position.z = z
    entry.pose.pose.orientation.z = math.sin(half)
    entry.pose.pose.orientation.w = math.cos(half)
    request = SetFreeJointState.Request()
    request.free_joints = [entry]
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=20.0)
    result = future.result()
    message = getattr(result, 'message', '') if result is not None else 'no response'
    if result is None or not getattr(result, 'success', False):
        raise SystemExit('set_free_joint_state failed: %s' % message)
    print('cube moved by MuJoCo to %.3f, %.3f, %.3f yaw %.0f deg'
          % (x, y, z, yaw_deg))
    node.destroy_node()


def main():
    distance = float(sys.argv[1]) if len(sys.argv) > 1 else 0.34
    lateral = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    # Yaw matters for the grasp: the jaws straddle two opposite faces, so a cube turned
    # 45 degrees presents its diagonal at them. Default 0 keeps the old behaviour.
    yaw_deg = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    rclpy.init()
    if os.environ.get('CCCSIM_BACKEND', 'gazebo') == 'mujoco':
        place_with_mujoco(distance, lateral, 0.057 / 2.0 + 0.001, yaw_deg)
        rclpy.shutdown()
        return

    xml = load_cube_generator().build_model_sdf(static=False)
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
