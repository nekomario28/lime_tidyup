#!/usr/bin/env python3
"""Spawn the robot with a camera sensor injected into its own URDF.

The robot's description has no camera and the RealSense Gazebo plugin cannot
find one, so the behaviour tree has no image to work with. A camera model placed
in the world is not a substitute: it stays where it was spawned and the robot
drives out of (or over) it. Moving a world model would need
libgazebo_ros_state.so, which segfaults gzserver in this world, so the camera is
attached to base_link instead — the image then follows the robot for free.

The sensors hang on the mount link from sim/camera_mount.py: a <pose> written
inside a <sensor> is dropped by the URDF->SDF conversion, so the offset has to be
a link. The same numbers drive sim/camera_tf.py, so images, depth and TF agree. A depth sensor rides alongside for the 3D cube centre: it
shares the pose and optics, so a box measured in the colour image indexes the
depth image without a separate registration step. It sits at the front edge of the base: further back the
robot's own body fills the frame and nothing is detected.
"""
from __future__ import annotations

import sys

import rclpy
from gazebo_msgs.srv import SpawnEntity
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

ENTITY = 'turtlebot3_lime_system'
CAMERA_XML = """
  <gazebo reference="sim_camera_mount">
    <sensor name="sim_color" type="camera">
      <always_on>true</always_on>
      <update_rate>15</update_rate>
      <visualize>false</visualize>
      <camera>
        <horizontal_fov>1.5</horizontal_fov>
        <image><width>848</width><height>480</height><format>R8G8B8</format></image>
        <clip><near>0.05</near><far>10.0</far></clip>
      </camera>
      <plugin name="sim_color_ros" filename="libgazebo_ros_camera.so">
        <ros><namespace>/camera/camera</namespace></ros>
        <camera_name>color</camera_name>
        <frame_name>sim_camera_color_optical_frame</frame_name>
      </plugin>
    </sensor>
    <sensor name="sim_depth" type="depth">
      <always_on>true</always_on>
      <update_rate>15</update_rate>
      <!-- Same link and optics as the colour sensor, so a box found in the
           colour image indexes the depth image directly. -->
      <visualize>false</visualize>
      <camera>
        <horizontal_fov>1.5</horizontal_fov>
        <image><width>848</width><height>480</height></image>
        <clip><near>0.05</near><far>10.0</far></clip>
      </camera>
      <plugin name="sim_depth_ros" filename="libgazebo_ros_camera.so">
        <ros><namespace>/camera/camera</namespace></ros>
        <camera_name>depth_cam</camera_name>
        <frame_name>sim_camera_depth_optical_frame</frame_name>
        <!-- The plugin clamps depth to 0.4 m by default regardless of the SDF
             near clip (measured: rows past the near field read 0 while the SDF
             said near_clip 0.05). A grasp happens at 0.4 m, so the range has to
             come from the plugin. -->
        <min_depth>0.05</min_depth>
        <max_depth>10.0</max_depth>
      </plugin>
    </sensor>
  </gazebo>
"""


class Fetcher(Node):
    def __init__(self):
        super().__init__('robot_spawn')
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE)
        self.description = None
        self.create_subscription(String, '/robot_description', self.on_description, qos)

    def on_description(self, msg):
        self.description = msg.data


def main():
    rclpy.init()
    fetcher = Fetcher()
    for _ in range(600):  # up to 60 s
        rclpy.spin_once(fetcher, timeout_sec=0.1)
        if fetcher.description:
            break
    if not fetcher.description:
        raise SystemExit('no /robot_description: is robot_state_publisher running?')

    urdf = fetcher.description
    if 'sim_color' in urdf:
        raise SystemExit('robot description already carries the sim camera')
    if '</robot>' not in urdf:
        raise SystemExit('unexpected robot description (no </robot>)')
    urdf = urdf.replace('</robot>', CAMERA_XML + '</robot>', 1)

    client = fetcher.create_client(SpawnEntity, '/spawn_entity')
    if not client.wait_for_service(timeout_sec=120.0):
        raise SystemExit('/spawn_entity unavailable')
    request = SpawnEntity.Request()
    request.name = ENTITY
    request.xml = urdf
    request.robot_namespace = ''
    request.initial_pose.position.x = 0.0
    request.initial_pose.position.y = 0.0
    request.initial_pose.position.z = 0.01
    request.initial_pose.orientation.w = 1.0
    request.reference_frame = 'world'
    future = client.call_async(request)
    rclpy.spin_until_future_complete(fetcher, future, timeout_sec=180.0)
    result = future.result()
    if result is None or not result.success:
        raise SystemExit(f'spawn failed: {result}')
    print(f'spawned {ENTITY} with a camera on base_link')


if __name__ == '__main__':
    sys.exit(main())
