#!/usr/bin/env python3
"""robot_state_publisher with use_sim:=true.

The upstream `rsp.launch.py` calls xacro without any mappings, so `use_sim`
falls back to false and `gazebo_ros2_control` is handed the *hardware* plugin
class (`turtlebot3_lime_hardware/...`), which does not exist in a sim session.
Controllers then never load and the arm cannot move. Passing the mapping here is
the whole fix; nothing upstream is modified.
"""
import os

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()
    publish_frequency = LaunchConfiguration('publish_frequency')
    ld.add_action(DeclareLaunchArgument('publish_frequency', default_value='15.0'))

    description = os.path.join(
        get_package_share_directory('turtlebot3_lime_description'),
        'urdf', 'turtlebot3_lime.urdf.xacro',
    )
    robot_description = xacro.process_file(
        description,
        mappings={'use_sim': 'true', 'use_fake_hardware': 'false',
                  'fake_sensor_commands': 'false', 'prefix': ''},
    ).toxml()
    print('[rsp_sim] hardware plugin:', 'gazebo_ros2_control/GazeboSystem'
          if 'gazebo_ros2_control/GazeboSystem' in robot_description
          else 'MISSING')

    ld.add_action(Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        respawn=True, output='screen',
        parameters=[{'publish_frequency': publish_frequency},
                    {'robot_description': robot_description}],
    ))
    return ld
