#!/usr/bin/env python3
"""Start MuJoCo's ros2_control node for the Lime grasp stack.

This mirrors the package's own demo, including the one thing that has to be right on
Humble: the control node reads the robot description from its *private*
`~/robot_description` topic, and `robot_state_publisher` publishes on `/robot_description`,
so the two are bridged by a remap here rather than with a `-r` argument on the command
line (where `~/` is not expanded to the node's namespace and the rule silently misses).

The physics comes from the scene named in the description's `mujoco_model` parameter --
sim/rsp_sim.launch.py puts it there when CCCSIM_BACKEND=mujoco -- and the plugins (/cmd_vel,
and later the camera) are configured by the parameter file next to this one.
"""
from __future__ import annotations

import os

from launch import LaunchDescription
from launch_ros.actions import Node

HERE = os.path.dirname(os.path.abspath(__file__))
PARAMETERS = [os.path.join(HERE, 'controllers_mujoco.yaml'),
              os.path.join(HERE, 'plugins_mujoco.yaml')]


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='mujoco_ros2_control',
            executable='ros2_control_node',
            output='both',
            parameters=[{'use_sim_time': True}] + PARAMETERS,
            remappings=[('~/robot_description', '/robot_description')],
        ),
    ])
