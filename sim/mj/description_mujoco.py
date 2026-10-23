#!/usr/bin/env python3
"""Turn the Gazebo robot description into the one MuJoCo's ros2_control interface wants.

The physics comes from the scene MJCF (sim/mj/build_scene.py); the URDF only has to
describe the ros2_control contract, and it has to describe one MuJoCo can honour:

- **The hardware plugin** becomes `mujoco_ros2_control/MujocoSystemInterface` and gains the
  parameters it reads: the scene file, headless, and the sim speed factor.
- **The wheel joints go.** They declare velocity command interfaces, and the interface
  documents the combinations it supports: a velocity command needs a `velocity` or
  `intvelocity` actuator in the MJCF, and a `position` actuator cannot serve it. Rather
  than add wheel actuators, `/cmd_vel` is served by the `BaseVelocityPlugin`, which
  overrides the base's free-joint velocity directly -- which is also why the wheels need
  no actuators at all.
- **The imu and battery sensors go**, for the same reason: a ros2_control sensor needs a
  matching MJCF sensor element, and the scene declares none. Nothing in the grasp path
  reads them (`imu_broadcaster` is not spawned either).

Everything else -- arm joints, the gripper joint, their limits -- is left exactly as the
Gazebo path has it, because those are the declarations the controllers are configured
against.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

GAZEBO_PLUGIN = 'gazebo_ros2_control/GazeboSystem'
MUJOCO_PLUGIN = 'mujoco_ros2_control/MujocoSystemInterface'
#: Joints and sensors the MuJoCo scene has no actuator/sensor for.
UNSUPPORTED_JOINTS = ('wheel_left_joint', 'wheel_right_joint', 'gripper_right_joint')
UNSUPPORTED_SENSORS = ('imu', 'battery')


def _drop_element(text, tag, name):
    pattern = r'\s*<%s name="%s">.*?</%s>' % (tag, re.escape(name), tag)
    return re.sub(pattern, '', text, flags=re.S)


def to_mujoco(description, scene_path, headless=True, sim_speed_factor=1.0):
    """Return the description with the MuJoCo hardware and its contract."""
    if GAZEBO_PLUGIN not in description:
        raise SystemExit('description does not carry %s; is it built with use_sim:=true?'
                         % GAZEBO_PLUGIN)
    parameters = ('<plugin>%s</plugin>\n'
                  '      <param name="mujoco_model">%s</param>\n'
                  '      <param name="headless">%s</param>\n'
                  '      <param name="sim_speed_factor">%s</param>'
                  % (MUJOCO_PLUGIN, scene_path,
                     'true' if headless else 'false', sim_speed_factor))
    description = description.replace('<plugin>%s</plugin>' % GAZEBO_PLUGIN, parameters)

    for joint in UNSUPPORTED_JOINTS:
        description = _drop_element(description, 'joint', joint)
    for sensor in UNSUPPORTED_SENSORS:
        description = _drop_element(description, 'sensor', sensor)
    return description


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--urdf', required=True)
    parser.add_argument('--scene', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--headless', action='store_true', default=True)
    arguments = parser.parse_args()

    text = Path(arguments.urdf).read_text()
    converted = to_mujoco(text, arguments.scene, headless=arguments.headless)
    Path(arguments.out).write_text(converted)
    print('wrote %s' % arguments.out)
    print('plugin: %s' % MUJOCO_PLUGIN)
    block = converted.split('<ros2_control', 1)[-1]
    for joint in UNSUPPORTED_JOINTS:
        head = block.split('name="%s"' % joint)
        still_there = len(head) > 1 and 'command_interface' in head[1][:400]
        print('  joint %-20s %s' % (joint, 'still declared' if still_there else 'removed'))


if __name__ == '__main__':
    main()
